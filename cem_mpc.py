"""
CEM-MPC controller for the nanochemistry transfer stage using the LeWM
trained in lewm_taesd.ipynb.

Given a goal camera frame, the planner captures the current state from the live
camera, rolls the latent world model (TAESD → ViT → LDP) forward under candidate
action sequences sampled by CEM, picks the sequence that minimises embedding-space
distance to the goal, executes the first action on the robot, then re-plans (MPC loop).

Usage:
    python cem_mpc.py --goal goal.png --ckpt checkpoints/jepa_epoch_0360.pt
    python cem_mpc.py --goal goal.png --dry_run   # plan only, no robot movement

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
from diffusers import AutoencoderTiny
from transformers import ViTModel, ViTConfig
from torch import nn

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from ldp import LatentDeltaPredictor
from module import MLP


# ─────────────────────────────────────────────────────────────────────────────
# Model definition (must match lewm_taesd.ipynb cell-model exactly)
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224
EMB_DIM  = 192

DEBOUNCE = 20   # loop iterations to skip after issuing a move


class FourierDeltaEmbedder(nn.Module):
    """Raw (Δx, Δy, Δz) in mm → EMB_DIM via random Fourier features."""
    def __init__(self, emb_dim=192, num_freqs=256, freq_scale=20.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(3, num_freqs) * freq_scale)
        self.proj = nn.Sequential(
            nn.Linear(3 * 2 * num_freqs, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),            nn.LayerNorm(emb_dim),
        )

    def forward(self, delta_xyz: torch.Tensor) -> torch.Tensor:
        x = delta_xyz.float().unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, 3, F)
        x = torch.cat([x.sin(), x.cos()], dim=-1).reshape(delta_xyz.shape[0], -1)
        return self.proj(x)


def build_model(device: str):
    vit_cfg = ViTConfig(
        num_channels=4, image_size=28, patch_size=4,
        hidden_size=EMB_DIM, num_hidden_layers=6,
        num_attention_heads=3, intermediate_size=768,
    )
    vae            = AutoencoderTiny.from_pretrained('madebyollin/taesd').to(device)
    encoder        = ViTModel(vit_cfg, add_pooling_layer=False).to(device)
    projector      = MLP(input_dim=EMB_DIM, hidden_dim=2048, output_dim=EMB_DIM,
                         norm_fn=nn.LayerNorm).to(device)
    delta_embedder = FourierDeltaEmbedder(emb_dim=EMB_DIM).to(device)
    ldp            = LatentDeltaPredictor(emb_dim=EMB_DIM, hidden_dim=512, depth=3).to(device)

    vae.requires_grad_(False)
    for m in [vae, encoder, projector, delta_embedder, ldp]:
        m.eval()

    return vae, encoder, projector, delta_embedder, ldp


def load_checkpoint(ckpt_path: Path, encoder, projector, delta_embedder, ldp, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    projector.load_state_dict(ckpt['projector'])
    delta_embedder.load_state_dict(ckpt['delta_embedder'])
    ldp.load_state_dict(ckpt['ldp'])
    print(f'Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get("epoch", "?")})')


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def frame_to_embedding(frame_rgb: np.ndarray, vae, encoder, projector,
                        device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, EMB_DIM) embedding."""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
    lat = vae.encoder(t)
    cls = encoder(lat, interpolate_pos_encoding=False).last_hidden_state[:, 0]
    return projector(cls)


# ─────────────────────────────────────────────────────────────────────────────
# CEM planner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cem_plan(
    current_emb:  torch.Tensor,    # (1, D)
    goal_emb:     torch.Tensor,    # (1, D)
    delta_embedder,
    ldp,
    *,
    horizon:      int   = 10,
    n_samples:    int   = 512,
    n_elite:      int   = 64,
    n_iters:      int   = 5,
    action_std:   float = 3.0,
    action_min:   float = -1e-2,
    action_max:   float =  1e-2,
    device:       str   = 'cpu',
) -> tuple[torch.Tensor, float, torch.Tensor]:
    mu  = torch.zeros(horizon, 3, device=device)
    std = torch.full((horizon, 3), action_std, device=device)

    for _ in range(n_iters):
        noise   = torch.randn(n_samples, horizon, 3, device=device)
        actions = (mu + std * noise).clamp(action_min, action_max)

        emb = current_emb.expand(n_samples, -1)
        for t in range(horizon):
            delta_emb = delta_embedder(actions[:, t])
            emb       = ldp(emb, delta_emb)

        costs = (emb - goal_emb.expand(n_samples, -1)).pow(2).sum(dim=-1)

        elite_idx     = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]
        mu  = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.1)

    return mu, costs[elite_idx[0]].item(), elite_actions.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame_bgr: np.ndarray, step: int, dist: float,
                 action: np.ndarray | None, cost: float | None) -> np.ndarray:
    out = frame_bgr.copy()
    lines = [f'Step: {step}', f'Dist: {dist:.4f}']
    if action is not None:
        lines.append(f'dx={action[0]:+.3f} dy={action[1]:+.3f} dz={action[2]:+.3f} mm')
    if cost is not None:
        lines.append(f'CEM cost: {cost:.4f}')
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CEM-MPC transfer stage controller.')
    p.add_argument('--goal',      required=True,       help='Path to goal frame image')
    p.add_argument('--ckpt',      default=None,        help='Path to JEPA checkpoint (.pt). '
                                                            'Defaults to latest in checkpoints/')
    p.add_argument('--dry_run',   action='store_true', help='Plan with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--horizon',   type=int,   default=10)
    p.add_argument('--samples',   type=int,   default=512)
    p.add_argument('--elite',     type=int,   default=64)
    p.add_argument('--iters',     type=int,   default=5)
    p.add_argument('--std',       type=float, default=3.0,  help='Initial action std (mm)')
    p.add_argument('--max_step',  type=float, default=10.0, help='Max Δ per axis per step (mm)')
    p.add_argument('--threshold', type=float, default=2.0,  help='Goal embedding dist threshold')
    p.add_argument('--settle',    type=float, default=0.5,  help='Settle time after move (s)')
    p.add_argument('--debounce',  type=int,   default=DEBOUNCE,
                                              help='Loop iterations to skip after a move')
    return p.parse_args()


def main():
    ACTION_SCALE = 0.01
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
    vae, encoder, projector, delta_embedder, ldp = build_model(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        candidates = sorted(Path('checkpoints').glob('jepa_epoch_*.pt'))
        if not candidates:
            raise FileNotFoundError('No jepa_epoch_*.pt checkpoints found in checkpoints/.')
        ckpt_path = candidates[-1]

    load_checkpoint(ckpt_path, encoder, projector, delta_embedder, ldp, device)

    # ── goal embedding ─────────────────────────────────────────────────────────
    goal_bgr = cv2.imread(args.goal)
    if goal_bgr is None:
        raise FileNotFoundError(f'Cannot read goal image: {args.goal}')
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    goal_emb = frame_to_embedding(goal_rgb, vae, encoder, projector, device)
    print(f'Goal embedding norm: {goal_emb.norm().item():.3f}')

    # ── hardware ──────────────────────────────────────────────────────────────
    cam   = None
    robot = None

    try:
        if not args.dry_run:
            from hardware.camera_controller import CameraController
            cam = CameraController(index=0, fps=15)
            cam.start()

        if not args.dry_run and not args.no_motion:
            from hardware.transfer_control_controller import TransferControl
            robot = TransferControl(only_xyz=True)
            for ax in AXES:
                robot.set_kst_speed(ax, max_vel=10.0, accel=1000.0, min_vel=0.0)
            print('Robot connected.')

        if args.no_motion:
            print('No-motion mode — camera live, moves skipped.')

        # ── cv2 loop ──────────────────────────────────────────────────────────
        cv2.namedWindow('CEM-MPC', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CEM-MPC', IMG_SIZE * 5, IMG_SIZE * 5)

        cumulative  = {ax: 0.0 for ax in AXES}
        step        = 0
        debounce_i  = 0          # counts down to 0 after each move
        dist        = float('inf')
        last_action = None
        last_cost   = None

        while True:
            # ── grab frame & display (every iteration) ────────────────────────
            if cam is not None:
                frame_bgr = cam.snap()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                frame_bgr = frame_rgb.copy()

            display = draw_overlay(frame_bgr, step, dist, last_action, last_cost)
            cv2.imshow('CEM-MPC', display)
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

            # ── observe & plan (only when ready) ──────────────────────────────
            current_emb = frame_to_embedding(frame_rgb, vae, encoder, projector, device)
            dist = (current_emb - goal_emb).norm().item()
            print(f'\nStep {step+1}  |  dist: {dist:.4f}')

            if dist < args.threshold:
                print('Goal reached.')
                break

            # ── plan ──────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            best_actions, best_cost, _ = cem_plan(
                current_emb, goal_emb, delta_embedder, ldp,
                horizon=args.horizon, n_samples=args.samples,
                n_elite=args.elite, n_iters=args.iters,
                action_std=args.std,
                action_min=-args.max_step, action_max=args.max_step,
                device=device,
            )
            print(f'  CEM {(time.perf_counter()-t0)*1000:.0f} ms  cost: {best_cost:.4f}')

            raw_action  = best_actions[0].cpu().numpy()
            last_action = raw_action * ACTION_SCALE
            last_cost   = best_cost

            # ── execute ───────────────────────────────────────────────────────
            if robot is not None:
                for ax, delta in zip(AXES, last_action):
                    print(f'  move {ax} by {delta:+.4f} mm')
                    robot.move_axis_by(ax, float(delta), timeout_ms=0)
                    cumulative[ax] += float(delta)
                time.sleep(args.settle)
                debounce_i = args.debounce
            else:
                for ax, delta in zip(AXES, last_action):
                    print(f'  move {ax} by {delta:+.4f} mm  (no-op)')

            step += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')

    finally:
        if robot is not None:
            print('Homing robot to start position...')
            for ax, total in cumulative.items():
                if total != 0.0:
                    print(f'  return {ax} by {-total:+.3f} mm')
                    robot.move_axis_by(ax, -total)
            robot.disconnect()
            print('Done.')
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
