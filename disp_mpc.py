"""
Displacement-predictor MPC controller for the nanochemistry transfer stage,
using the Siamese-ViT displacement model trained in displacement_pred.ipynb.

Unlike cem_mpc.py (which rolls a latent world model forward under sampled
action sequences), this controller directly regresses the (Δx, Δy, Δz)
displacement between the current camera frame and the goal frame in a single
forward pass — no planning/rollout required. Each MPC step: observe → predict
Δxyz → move a scaled fraction of it → re-observe.

Usage:
    python disp_mpc.py --goal goal.png --ckpt checkpoints/disp_epoch_0080.pt
    python disp_mpc.py --goal goal.png --dry_run   # predict only, no robot movement

All distances are in mm (the unit used during training).
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

IMG_SIZE = 224
EMB_DIM  = 192
PX_PER_MM = 8.0

DEBOUNCE = 200   # loop iterations to skip after issuing a move


# ─────────────────────────────────────────────────────────────────────────────
# Model definition (must match displacement_pred.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────

class DisplacementHead(nn.Module):
    """[emb_a, emb_b, emb_a − emb_b] → (3,) normalised displacement."""
    def __init__(self, emb_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * emb_dim),
            nn.Linear(3 * emb_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 3),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([emb_a, emb_b, emb_a - emb_b], dim=-1))


def build_model(device: str):
    vit_cfg = ViTConfig(
        num_channels=3, image_size=IMG_SIZE, patch_size=16,
        hidden_size=EMB_DIM, num_hidden_layers=6,
        num_attention_heads=3, intermediate_size=768,
    )
    encoder = ViTModel(vit_cfg, add_pooling_layer=False).to(device)
    head    = DisplacementHead(emb_dim=EMB_DIM).to(device)
    encoder.eval()
    head.eval()
    return encoder, head


def _remap_encoder_keys(sd: dict) -> dict:
    """Remap checkpoint keys from older transformers naming to current ViTModel naming.

    Older: layers.N.attention.q_proj / k_proj / v_proj / o_proj / mlp.fc1 / mlp.fc2
    Current: encoder.layer.N.attention.attention.query / key / value /
             attention.output.dense / intermediate.dense / output.dense
    """
    import re
    subst = [
        ('attention.q_proj',    'attention.attention.query'),
        ('attention.k_proj',    'attention.attention.key'),
        ('attention.v_proj',    'attention.attention.value'),
        ('attention.o_proj',    'attention.output.dense'),
        ('mlp.fc1',             'intermediate.dense'),
        ('mlp.fc2',             'output.dense'),
    ]
    new_sd = {}
    for k, v in sd.items():
        new_k = re.sub(r'^layers\.(\d+)\.', r'encoder.layer.\1.', k)
        for old, new in subst:
            new_k = new_k.replace(old, new)
        new_sd[new_k] = v
    return new_sd


def load_checkpoint(ckpt_path: Path, encoder, head, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc_sd = ckpt['encoder']
    if any(k.startswith('layers.') for k in enc_sd):
        enc_sd = _remap_encoder_keys(enc_sd)
    encoder.load_state_dict(enc_sd)
    head.load_state_dict(ckpt['head'])
    delta_mean = ckpt['delta_mean'].to(device)
    delta_std  = ckpt['delta_std'].to(device)
    print(f'Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get("epoch", "?")})')
    return delta_mean, delta_std


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frame(frame_rgb: np.ndarray, encoder, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, EMB_DIM) CLS embedding."""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
    return encoder(t, interpolate_pos_encoding=False).last_hidden_state[:, 0]


@torch.no_grad()
def predict_displacement(cur_emb: torch.Tensor, goal_emb: torch.Tensor, head,
                         delta_mean: torch.Tensor, delta_std: torch.Tensor) -> np.ndarray:
    """Predicted (Δx, Δy, Δz) in mm to move from cur → goal."""
    pred_norm = head(cur_emb, goal_emb)
    pred_mm   = pred_norm * delta_std + delta_mean
    return pred_mm.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

ARROW_MIN_PX = 40    # arrow always at least this long when direction is nonzero
ARROW_MAX_PX = 150   # cap so it never blows past the frame


def draw_overlay(frame_bgr: np.ndarray, step: int, dist: float,
                 pred_xyz: np.ndarray | None, move_xyz: np.ndarray | None,
                 frame_counter: int) -> np.ndarray:
    out  = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # intended x/y move vector as an arrow from frame center.
    # Direction is exact; length is visually rescaled (independent of the
    # tiny physical mm magnitude) so it's always readable on screen.
    if move_xyz is not None:
        dx, dy = float(move_xyz[0]), float(move_xyz[1])
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag > 1e-6:
            ux, uy   = dx / mag, dy / mag
            arrow_px = min(ARROW_MAX_PX, max(ARROW_MIN_PX, mag * PX_PER_MM))
            end = (int(cx + ux * arrow_px), int(cy - uy * arrow_px))
            cv2.arrowedLine(out, (cx, cy), end, (0, 140, 255), 4, tipLength=0.3)
    cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thick = 3

    lines = [f'Step: {step}', f'Dist: {dist:.4f} mm']
    if pred_xyz is not None:
        lines.append(f'pred dx={pred_xyz[0]:+.3f} dy={pred_xyz[1]:+.3f} dz={pred_xyz[2]:+.3f} mm')
    if move_xyz is not None:
        lines.append(f'move dx={move_xyz[0]:+.3f} dy={move_xyz[1]:+.3f} dz={move_xyz[2]:+.3f} mm')
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
    p = argparse.ArgumentParser(description='Displacement-predictor MPC transfer stage controller.')
    p.add_argument('--goal',      required=True,       help='Path to goal frame image')
    p.add_argument('--ckpt',      default=None,        help='Path to displacement checkpoint (.pt). '
                                                            'Defaults to latest disp_epoch_*.pt in checkpoints/')
    p.add_argument('--dry_run',   action='store_true', help='Predict with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--scale',     type=float, default=0.3,  help='Fraction of predicted Δ to execute per step')
    p.add_argument('--max_step',  type=float, default=1.0,  help='Max |Δ| per axis per step (mm)')
    p.add_argument('--threshold', type=float, default=0.3,  help='Goal distance threshold (mm)')
    p.add_argument('--settle',    type=float, default=0.5,  help='Settle time after move (s)')
    p.add_argument('--debounce',  type=int,   default=DEBOUNCE,
                                              help='Loop iterations to skip after a move')
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
    encoder, head = build_model(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        candidates = sorted(Path('checkpoints').glob('disp_epoch_*.pt'))
        if not candidates:
            raise FileNotFoundError('No disp_epoch_*.pt checkpoints found in checkpoints/.')
        ckpt_path = candidates[-1]

    delta_mean, delta_std = load_checkpoint(ckpt_path, encoder, head, device)

    # ── goal embedding ────────────────────────────────────────────────────────
    goal_bgr = cv2.imread(args.goal)
    if goal_bgr is None:
        raise FileNotFoundError(f'Cannot read goal image: {args.goal}')
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    goal_emb = encode_frame(goal_rgb, encoder, device)

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
            for ax in AXES:
                robot.set_kst_speed(ax, max_vel=10.0, accel=10000.0, min_vel=0.0)
            p = robot.positions()
            print('Robot connected. Positions:', p)
            position = [float(pos) for _, pos in robot.positions().items()]
            start_position = position.copy()

        if args.no_motion:
            print('No-motion mode — camera live, moves skipped.')

        # ── cv2 loop ──────────────────────────────────────────────────────────
        cv2.namedWindow('Disp-MPC', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Disp-MPC', IMG_SIZE * 5, IMG_SIZE * 5)

        step          = 0
        frame_counter = 0
        debounce_i    = 0
        dist          = float('inf')
        last_pred     = None   # (3,) full predicted displacement (mm)
        last_move     = None   # (3,) scaled/clamped move actually issued (mm)

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
            cv2.imshow('Disp-MPC', display)
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
            cur_emb   = encode_frame(frame_rgb, encoder, device)
            pred_xyz  = predict_displacement(cur_emb, goal_emb, head, delta_mean, delta_std)
            dist      = float(np.linalg.norm(pred_xyz))
            last_pred = pred_xyz
            print(f'\nStep {step+1}  |  dist: {dist:.4f} mm  |  pred: {pred_xyz}')

            if dist < args.threshold:
                print('Goal reached.')
                break

            # ── scale & clamp move ────────────────────────────────────────────
            move_xyz  = np.clip(pred_xyz * args.scale, -args.max_step, args.max_step)
            last_move = move_xyz

            # ── execute ───────────────────────────────────────────────────────
            if robot is not None:
                for i, ax in enumerate(AXES):
                    position[i] += float(move_xyz[i])
                    print(f'  move {ax} to {position[i]:+.4f} mm')
                    robot.move_axis_to(ax, position[i])
                debounce_i = args.debounce
            else:
                for ax, delta in zip(AXES, move_xyz):
                    print(f'  move {ax} by {delta:+.4f} mm  (no-op)')

            step += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')

    except Exception as e:
        print(f'Error: {e}')
        raise

    finally:
        if robot is not None and start_position:
            print('Homing robot to start position...')
            time.sleep(args.settle)
            for ax, pos in zip(AXES, start_position):
                print(f'  return {ax} to {pos:+.3f} mm')
                robot.move_axis_to(ax, pos)
                time.sleep(args.settle)
            robot.disconnect()
            print('Done.')
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
