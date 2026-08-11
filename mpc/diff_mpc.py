"""
Diff-predictor MPC controller for the nanochemistry transfer stage,
using the diff-pred ViT trained in diff_pred.ipynb.

Unlike disp_mpc.py's old Siamese-ViT (which encoded current/goal frames
separately and concatenated their embeddings), this model runs a single ViT
over the *blurred difference image* blur(goal) - blur(current) and regresses
[Δx, Δy, Δcx, Δcy, Δarea] from its CLS token in one forward pass. There is no
reliable *magnitude* Δz signal in this model, so z is not driven proportionally
like x/y. Instead, control runs in two phases:
  1. 'xy' — move by the predicted Δ(x, y) (mm) each step until within --threshold.
  2. 'z'  — once xy has converged, nudge z by a fixed --z_step each step: if the
            predicted target ring area is greater than the current ring area
            (Δarea > 0) move z down, otherwise move z up, until |Δarea| is within
            --area_threshold (px²).

Each MPC step: observe → predict → move (xy or z, depending on phase) → re-observe.

Usage:
    python diff_mpc.py --goal goal.png --ckpt checkpoints/diff_pred_epoch_0040.pt
    python diff_mpc.py --goal goal.png --dry_run   # predict only, no robot movement
    python diff_mpc.py --goal goal.png --record    # save video to ./demo.mp4

All distances are in mm (the unit used during training), except Δcx/Δcy/Δarea which are px.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import ViTModel, ViTConfig
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import hardware` resolves regardless of cwd

IMG_SIZE    = 224   # ViT input resolution — unrelated to the camera's native display resolution
PATCH_SIZE  = 16
EMB_DIM     = 192
BLUR_SIGMA  = 2.0    # must match diff_pred.ipynb training
DRY_RUN_FRAME_SIZE = (960, 540)   # placeholder frame size when --dry_run has no camera attached

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

def draw_overlay(frame_bgr: np.ndarray, step: int) -> np.ndarray:
    """Live camera feed (at its native, pre-resize resolution) with just the step count top-right."""
    out = frame_bgr.copy()
    w   = out.shape[1]

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thick = 3

    label = f'Step: {step}'
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(out, label, (w - tw - 10, th + 10), font, scale, (0, 220, 255), thick, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Diff-predictor MPC transfer stage controller.')
    p.add_argument('--goal',      required=True,       help='Path to goal frame image')
    p.add_argument('--ckpt',      default=str(REPO_ROOT / 'checkpoints' / 'diff_pred_epoch_0040.pt'),
                                              help='Path to diff-pred checkpoint (.pt). '
                                                   'Defaults to latest diff_pred_epoch_*.pt in checkpoints/')
    p.add_argument('--dry_run',   action='store_true', help='Predict with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--scale',     type=float, default=1,  help='Fraction of predicted Δ to execute per step')
    p.add_argument('--max_step',  type=float, default=0.1,  help='Max |Δ| per axis per step (mm)')
    p.add_argument('--threshold', type=float, default=0.05,  help='Goal xy distance threshold (mm)')
    p.add_argument('--area_threshold', type=float, default=500.0,
                                              help='Goal Newton-ring |Δarea| threshold (px²), checked after xy converges')
    p.add_argument('--z_step',    type=float, default=0.005, help='Fixed z nudge per step while aligning ring area (mm)')
    p.add_argument('--debounce',  type=int,   default=DEBOUNCE,
                                              help='Loop iterations to skip after a move')
    p.add_argument('--record',    action='store_true', help='Record displayed frames to ./demo.mp4')
    return p.parse_args()


def main():
    AXES = ('x', 'y', 'z')

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
        candidates = sorted((REPO_ROOT / 'checkpoints').glob('diff_pred_epoch_*.pt'))
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
    video_writer = None
    position = []

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

        if args.no_motion:
            print('No-motion mode — camera live, moves skipped.')

        # ── cv2 loop ──────────────────────────────────────────────────────────
        cv2.namedWindow('Diff-MPC', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Diff-MPC', DRY_RUN_FRAME_SIZE[0], DRY_RUN_FRAME_SIZE[1])

        step          = 0
        debounce_i    = 0
        dist          = float('inf')
        phase         = 'xy'   # 'xy' -> 'z'

        while True:
            if cam is not None:
                frame_bgr = cam.snap()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = np.zeros((DRY_RUN_FRAME_SIZE[1], DRY_RUN_FRAME_SIZE[0], 3), dtype=np.uint8)
                print(frame_rgb.shape)
                frame_bgr = frame_rgb.copy()

            display = draw_overlay(frame_bgr, step)
            if args.record:
                if video_writer is None:
                    height, width = display.shape[:2]
                    output_path = Path.cwd() / 'demo.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        15.0,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f'Cannot open video writer: {output_path}')
                    print(f'Recording video to {output_path}')

                video_writer.write(display)

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
            pred_area = float(pred_full[4])
            dist      = float(np.linalg.norm(pred_xy))

            if phase == 'xy':
                if dist < args.threshold:
                    print(f'xy converged ({dist:.4f} mm away) — switching to z alignment.')
                    phase = 'z'
                    continue

                # ── scale & clamp move (x, y only — no Δz magnitude signal) ───
                move_xy = np.clip(pred_xy * args.scale, -args.max_step, args.max_step)

                if robot is not None:
                    for i, ax in enumerate(('x', 'y')):
                        position[i] += float(move_xy[i])
                        robot.move_axis_to(ax, position[i])
                    print(f'  move x to {position[0]:+.4f} mm, y to {position[1]:+.4f} mm')
                    debounce_i = args.debounce
                else:
                    print(f'  move x by {move_xy[0]:+.4f} mm, y by {move_xy[1]:+.4f} mm  (no-op)')

            else:   # phase == 'z'
                if abs(pred_area) < args.area_threshold:
                    print(f'Goal reached. xy={dist:.4f} mm away, |Δarea|={abs(pred_area):.2f} px² away.')
                    break

                # target ring area greater than current -> move down; otherwise up.
                move_z = args.z_step if pred_area > 0 else -args.z_step

                if robot is not None:
                    position[2] += move_z
                    print(f'  move z to {position[2]:+.4f} mm  (Δarea={pred_area:+.2f} px²)')
                    robot.move_axis_to('z', position[2])
                    debounce_i = args.debounce
                else:
                    print(f'  move z by {move_z:+.4f} mm  (Δarea={pred_area:+.2f} px²)  (no-op)')

            step += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')

    except Exception as e:
        print(f'Error: {e}')
        raise

    finally:
        if robot is not None:
            robot.disconnect()
            print('Done.')
        if cam is not None:
            cam.stop()
        if video_writer is not None:
            video_writer.release()
            print(f'Video saved to {Path.cwd() / "demo.mp4"}')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
