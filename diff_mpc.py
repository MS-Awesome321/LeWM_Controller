"""
Diff-predictor MPC controller for the nanochemistry transfer stage,
using the diff-pred ViT trained in diff_pred.ipynb.

Unlike disp_mpc.py's old Siamese-ViT (which encoded current/goal frames
separately and concatenated their embeddings), this model runs a single ViT
over the *blurred difference image* blur(goal) - blur(current) and regresses
[Δx, Δy, Δcx, Δcy, Δarea] from its CLS token in one forward pass. Only Δx, Δy
(mm) are used to move the stage — Δcx/Δcy/Δarea (Newton-ring centroid/area,
px units) are printed/overlaid for diagnostics but there is no reliable Δz
signal in this model, so the z axis is never moved.

Each MPC step: observe → predict Δ(x, y) → move a scaled fraction of it → re-observe.

Usage:
    python diff_mpc.py --goal goal.png --ckpt checkpoints/diff_pred_epoch_0040.pt
    python diff_mpc.py --goal goal.png --dry_run   # predict only, no robot movement

All distances are in mm (the unit used during training), except Δcx/Δcy/Δarea which are px.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import ViTModel, ViTConfig
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

IMG_SIZE    = 224
PATCH_SIZE  = 16
EMB_DIM     = 192
BLUR_SIGMA  = 2.0    # must match diff_pred.ipynb training
PX_PER_MM   = 1e5

DEBOUNCE = 20   # loop iterations to skip after issuing a move


# ─────────────────────────────────────────────────────────────────────────────
# Model definition (must match diff_pred.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────

class DisplacementHead(nn.Module):
    """CLS token -> small MLP -> (5,): [Δx, Δy, Δcx, Δcy, Δarea]."""
    def __init__(self, emb_dim=EMB_DIM, target_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, target_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


def build_model(device: str):
    vit_cfg = ViTConfig(
        num_channels=3, image_size=IMG_SIZE, patch_size=PATCH_SIZE,
        hidden_size=EMB_DIM, num_hidden_layers=6,
        num_attention_heads=3, intermediate_size=768,
    )
    vit  = ViTModel(vit_cfg, add_pooling_layer=False).to(device)
    head = DisplacementHead(emb_dim=EMB_DIM).to(device)
    vit.eval()
    head.eval()
    return vit, head


def load_checkpoint(ckpt_path: Path, vit, head, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vit.load_state_dict(ckpt['vit'])
    head.load_state_dict(ckpt['head'])
    delta_mean = ckpt['delta_mean'].to(device)
    delta_std  = ckpt['delta_std'].to(device)
    print(f'Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get("epoch", "?")})')
    return delta_mean, delta_std


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gaussian_kernel(sigma: float, device: str):
    """(1, 1, k, k) normalized 2D gaussian kernel, k = 2*round(3*sigma)+1."""
    if sigma <= 0:
        return None
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])


def gaussian_blur(x: torch.Tensor, kernel: torch.Tensor | None) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H, W), blurred per-channel (depthwise conv, same padding)."""
    if kernel is None:
        return x
    c = x.shape[1]
    k = kernel.repeat(c, 1, 1, 1).to(dtype=x.dtype, device=x.device)
    pad = k.shape[-1] // 2
    return torch.nn.functional.conv2d(x, k, padding=pad, groups=c)


def to_float_frame(frame_rgb: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (1, 3, IMG_SIZE, IMG_SIZE) float32 [0, 1] on device."""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def predict_displacement(cur_rgb: np.ndarray, goal_rgb: np.ndarray, vit, head,
                         blur_kernel, delta_mean: torch.Tensor, delta_std: torch.Tensor,
                         device: str) -> np.ndarray:
    """Predicted [Δx, Δy, Δcx, Δcy, Δarea] (mm, mm, px, px, px²) to move from cur → goal."""
    cur  = to_float_frame(cur_rgb, device)
    goal = to_float_frame(goal_rgb, device)
    diff = gaussian_blur(goal, blur_kernel) - gaussian_blur(cur, blur_kernel)
    cls_token = vit(diff, interpolate_pos_encoding=False).last_hidden_state[:, 0]
    pred_norm = head(cls_token)
    pred_raw  = pred_norm * delta_std + delta_mean
    return pred_raw.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

ARROW_MIN_PX = 40    # arrow always at least this long when direction is nonzero
ARROW_MAX_PX = 1000   # cap so it never blows past the frame


def draw_overlay(frame_bgr: np.ndarray, step: int, dist: float,
                 pred_full: np.ndarray | None, move_xy: np.ndarray | None,
                 frame_counter: int) -> np.ndarray:
    out  = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # intended x/y move vector as an arrow from frame center.
    # Direction is exact; length is visually rescaled (independent of the
    # tiny physical mm magnitude) so it's always readable on screen.
    if move_xy is not None:
        dx, dy = float(move_xy[0]), float(move_xy[1])
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag > 1e-6:
            arrow_px = min(ARROW_MAX_PX, max(ARROW_MIN_PX, mag * PX_PER_MM))
            end = (int(cx + dx * arrow_px), int(cy - dy * arrow_px))
            cv2.arrowedLine(out, (cx, cy), end, (0, 140, 255), 4, tipLength=0.3)
    cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thick = 3

    lines = [f'Step: {step}', f'Dist: {dist:.4f} mm']
    if pred_full is not None:
        lines.append(f'pred dx={pred_full[0]:+.3f} dy={pred_full[1]:+.3f} mm')
        lines.append(f'pred dcx={pred_full[2]:+.2f} dcy={pred_full[3]:+.2f} darea={pred_full[4]:+.1f} px')
    if move_xy is not None:
        lines.append(f'move dx={move_xy[0]:+.3f} dy={move_xy[1]:+.3f} mm')
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (10, 44 + i * 44), font, scale, (0, 255, 0), thick, cv2.LINE_AA)

    label = f'Frame {frame_counter}'
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(out, label, (w - tw - 10, th + 10), font, scale, (0, 220, 255), thick, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Diff-predictor MPC transfer stage controller.')
    p.add_argument('--goal',      required=True,       help='Path to goal frame image')
    p.add_argument('--ckpt',      default='checkpoints/diff_pred_epoch_0040.pt', help='Path to diff-pred checkpoint (.pt). '
                                                                            'Defaults to latest diff_pred_epoch_*.pt in checkpoints/')
    p.add_argument('--dry_run',   action='store_true', help='Predict with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--scale',     type=float, default=1,  help='Fraction of predicted Δ to execute per step')
    p.add_argument('--max_step',  type=float, default=0.1,  help='Max |Δ| per axis per step (mm)')
    p.add_argument('--threshold', type=float, default=0.1,  help='Goal distance threshold (mm)')
    p.add_argument('--settle',    type=float, default=0.5,  help='Settle time after move (s)')
    p.add_argument('--debounce',  type=int,   default=DEBOUNCE,
                                              help='Loop iterations to skip after a move')
    return p.parse_args()


def main():
    AXES = ('x', 'y', 'z')   # z is never moved — the diff-pred model has no Δz signal

    args = parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── model ─────────────────────────────────────────────────────────────────
    vit, head = build_model(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        candidates = sorted(Path('checkpoints').glob('diff_pred_epoch_*.pt'))
        if not candidates:
            raise FileNotFoundError('No diff_pred_epoch_*.pt checkpoints found in checkpoints/.')
        ckpt_path = candidates[-1]

    delta_mean, delta_std = load_checkpoint(ckpt_path, vit, head, device)
    blur_kernel = make_gaussian_kernel(BLUR_SIGMA, device)

    # ── goal frame ────────────────────────────────────────────────────────────
    goal_bgr = cv2.imread(args.goal)
    if goal_bgr is None:
        raise FileNotFoundError(f'Cannot read goal image: {args.goal}')
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)

    # ── hardware ──────────────────────────────────────────────────────────────
    cam   = None
    robot = None
    position = []
    start_position = []

    try:
        if not args.dry_run:
            from hardware.camera_controller import CameraController
            cam = CameraController(index=0, fps=15)
            cam.start()

        if not args.dry_run and not args.no_motion:
            from hardware.transfer_control_controller import TransferControl
            robot = TransferControl(only_xyz=True)
            robot.connect()
            # for ax in AXES:
            #     robot.set_kst_speed(ax, max_vel=10.0, accel=10000.0, min_vel=0.0)
            p = robot.positions()
            print('Robot connected. Positions:', p)
            position = [float(pos) for _, pos in robot.positions().items()]
            start_position = position.copy()

        if args.no_motion:
            print('No-motion mode — camera live, moves skipped.')

        # ── cv2 loop ──────────────────────────────────────────────────────────
        cv2.namedWindow('Diff-MPC', cv2.WINDOW_NORMAL)

        step          = 0
        frame_counter = 0
        debounce_i    = 0
        dist          = float('inf')
        last_pred     = None   # (5,) full predicted [Δx, Δy, Δcx, Δcy, Δarea]
        last_move     = None   # (2,) scaled/clamped (Δx, Δy) move actually issued (mm)
        goal_reached = False

        while True:
            # ── grab frame & display (every iteration) ────────────────────────
            if cam is not None:
                frame_bgr = cam.snap()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                frame_bgr = frame_rgb.copy()

            frame_counter += 1
            display = draw_overlay(frame_bgr, step, dist, last_pred, last_move, frame_counter)
            cv2.imshow('Diff-MPC', display)
            if cv2.waitKey(1) == 27:   # ESC
                print('ESC — stopping.')
                break

            # ── debounce ──────────────────────────────────────────────────────
            if debounce_i > 0:
                debounce_i -= 1
                continue

            if robot is not None:
                kst_axes = [robot._get_axis(ax) for ax in AXES]
                if any(a.dev.Status.IsMoving for a in kst_axes):
                    continue

            # ── observe & predict (only when ready) ───────────────────────────
            pred_full = predict_displacement(frame_rgb, goal_rgb, vit, head, blur_kernel,
                                             delta_mean, delta_std, device)
            pred_xy   = pred_full[:2]
            dist      = float(np.linalg.norm(pred_xy))
            last_pred = pred_full
            # print(f'\nStep {step+1}  |  dist: {dist:.4f} mm  |  pred xy: {pred_xy}')

            if dist < args.threshold:
                print(f'Goal reached. {dist} mm away.')
                goal_reached = True
                break

            # ── scale & clamp move (x, y only — no Δz signal in this model) ───
            move_xy   = np.clip(pred_xy * args.scale, -args.max_step, args.max_step)
            last_move = move_xy

            # ── execute ───────────────────────────────────────────────────────
            if robot is not None:
                for i, ax in enumerate(('x', 'y')):
                    position[i] += float(move_xy[i])
                    print(f'  move {ax} to {position[i]:+.4f} mm')
                    robot.move_axis_to(ax, position[i])
                debounce_i = args.debounce
            else:
                for ax, delta in zip(('x', 'y'), move_xy):
                    print(f'  move {ax} by {delta:+.4f} mm  (no-op)')

            step += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')

    except Exception as e:
        print(f'Error: {e}')
        raise

    finally:
        if start_position and not goal_reached:
            print('Homing robot to start position...')
            robot.stop_xyz()
            for ax, pos in zip(AXES, start_position):
                print(f'  return {ax} to {pos:+.3f} mm')
                robot.move_axis_to(ax, pos)

            time.sleep(args.settle)
            if cam is not None:
                frame_bgr = cam.snap()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                frame_bgr = frame_rgb.copy()

            cv2.imshow('Diff-MPC', frame_bgr)
            cv2.waitKey(1000)

        if robot is not None:
            robot.disconnect()
            print('Done.')
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
