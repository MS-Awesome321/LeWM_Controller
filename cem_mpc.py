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
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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


def load_checkpoint(ckpt_path: Path, encoder, projector, delta_embedder, ldp, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc_sd = ckpt['encoder']
    # detect old key scheme and remap if necessary
    # if any(k.startswith('layers.') for k in enc_sd):
    #     enc_sd = _remap_encoder_keys(enc_sd)
    encoder.load_state_dict(enc_sd)
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
    lat = vae.encoder(t)                                               # (1, 4, H_lat, W_lat)
    cls = encoder(lat, interpolate_pos_encoding=False).last_hidden_state[:, 0]
    return projector(cls)                                              # (1, EMB_DIM)


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
    action_std:   float = 3.0,     # mm — initial sampling std
    action_min:   float = -1e-2,   # mm — per-step clamp
    action_max:   float =  1e-2,
    device:       str   = 'cpu',
) -> tuple[torch.Tensor, float]:
    """
    Cross-Entropy Method over (Δx, Δy, Δz) action sequences.

    Returns:
        best_actions  : (horizon, 3) tensor — planned action sequence in mm
        best_cost     : scalar — final embedding distance to goal
        elite_actions : (n_elite, horizon, 3) tensor — all elite sequences
    """
    mu  = torch.zeros(horizon, 3, device=device)
    std = torch.full((horizon, 3), action_std, device=device)

    for _ in range(n_iters):
        # sample K action sequences: (K, H, 3)
        noise   = torch.randn(n_samples, horizon, 3, device=device)
        actions = (mu + std * noise).clamp(action_min, action_max)

        # roll out world model for all K sequences in parallel
        emb = current_emb.expand(n_samples, -1)         # (K, D)
        for t in range(horizon):
            delta_emb = delta_embedder(actions[:, t])   # (K, D)
            emb       = ldp(emb, delta_emb)             # (K, D)

        # cost: L2 distance to goal in embedding space at the final step
        costs = (emb - goal_emb.expand(n_samples, -1)).pow(2).sum(dim=-1)  # (K,)

        # refit distribution from elite samples
        elite_idx     = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]              # (n_elite, H, 3)
        mu  = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.1)

    best_cost = costs[elite_idx[0]].item()
    return mu, best_cost, elite_actions.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(cam) -> np.ndarray:
    """Returns (H, W, 3) uint8 RGB from a CameraController instance."""
    bgr = cam.snap()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def send_overlay(proc: subprocess.Popen, best_actions: torch.Tensor,
                 elite_actions: torch.Tensor, dist: float, step: int) -> None:
    """Serialise plan data as a single JSON line and write to liveview's stdin."""
    if proc is None or proc.poll() is not None:
        return
    payload = json.dumps({
        'step':  step,
        'dist':  dist,
        'best':  best_actions.cpu().tolist(),
        'elite': elite_actions.cpu().tolist(),
    }, separators=(',', ':'))
    try:
        proc.stdin.write((payload + '\n').encode())
        proc.stdin.flush()
    except BrokenPipeError:
        pass


def run_mpc(
    goal_image:      np.ndarray,         # (H, W, 3) uint8 RGB
    vae, encoder, projector, delta_embedder, ldp,
    robot,                               # TransferControl instance (or None)
    cam,                                 # CameraController instance (or None for dry-run)
    liveview_proc:   subprocess.Popen | None,
    *,
    device:          str   = 'cpu',
    max_steps:       int   = 50,
    goal_threshold:  float = 2.0,
    horizon:         int   = 10,
    n_samples:       int   = 512,
    n_elite:         int   = 64,
    n_iters:         int   = 5,
    action_std:      float = 3.0,
    action_min:      float = -10.0,
    action_max:      float =  10.0,
    action_scale:    float = 1.0,
    axes:            tuple = ('x', 'y', 'z'),
    settle_s:        float = 0.5,
) -> dict:
    """
    MPC loop: capture frame → plan → send overlay to liveview → execute → repeat.

    Returns cumulative displacement per axis so the caller can home on interrupt.
    cam=None triggers dry-run (black frame, no robot moves).
    """
    goal_emb = frame_to_embedding(goal_image, vae, encoder, projector, device)
    print(f'Goal embedding norm: {goal_emb.norm().item():.3f}')

    cumulative = {ax: 0.0 for ax in axes}

    for step in range(max_steps):
        # ── observe ───────────────────────────────────────────────────────────
        frame = capture_frame(cam) if cam is not None else \
                np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        current_emb = frame_to_embedding(frame, vae, encoder, projector, device)
        dist = (current_emb - goal_emb).norm().item()
        print(f'\nStep {step+1}/{max_steps}  |  embedding dist to goal: {dist:.4f}')

        if dist < goal_threshold:
            print('Goal reached.')
            break

        # ── plan ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        best_actions, best_cost, elite_actions = cem_plan(
            current_emb, goal_emb, delta_embedder, ldp,
            horizon=horizon, n_samples=n_samples, n_elite=n_elite,
            n_iters=n_iters, action_std=action_std,
            action_min=action_min, action_max=action_max, device=device,
        )
        plan_ms = (time.perf_counter() - t0) * 1000
        print(f'  CEM done in {plan_ms:.0f} ms  |  predicted final cost: {best_cost:.4f}')

        # send plan to liveview subprocess — it draws while we move
        send_overlay(liveview_proc, best_actions, elite_actions, dist, step + 1)

        # ── execute first action ───────────────────────────────────────────────
        raw_action = best_actions[0].cpu().numpy()   # (3,) [Δx, Δy, Δz] mm
        action     = raw_action * action_scale

        if robot is not None:
            for ax, delta in zip(axes, action):
                print(f'  move {ax} by {delta:+.3f} mm  (raw {raw_action[list(axes).index(ax)]:+.3f} × {action_scale})')
                robot.move_axis_by(ax, float(delta), timeout_ms=0)
                cumulative[ax] += float(delta)
            # wait for all axes to finish moving
            t_settle = time.perf_counter()
            kst_axes = [robot._get_axis(ax) for ax in axes if ax in ('x', 'y', 'z')]
            while any(a.dev.Status.IsMoving for a in kst_axes):
                if time.perf_counter() - t_settle > settle_s + 10.0:
                    break
                time.sleep(0.02)
            time.sleep(settle_s)
        else:
            for ax, delta in zip(axes, action):
                print(f'  move {ax} by {delta:+.3f} mm  (no-op)')

    else:
        print('Max steps reached without converging.')

    return cumulative


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CEM-MPC transfer stage controller.')
    p.add_argument('--goal',     required=True,  help='Path to goal frame image')
    p.add_argument('--ckpt',     default=None,   help='Path to JEPA checkpoint (.pt). '
                                                       'Defaults to latest in checkpoints/')
    p.add_argument('--dry_run',   action='store_true', help='Plan with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--horizon',  type=int,   default=10)
    p.add_argument('--samples',  type=int,   default=512)
    p.add_argument('--elite',    type=int,   default=64)
    p.add_argument('--iters',    type=int,   default=5)
    p.add_argument('--std',      type=float, default=3.0,  help='Initial action std (mm)')
    p.add_argument('--max_step', type=float, default=10.0, help='Max Δ per axis per step (mm)')
    p.add_argument('--steps',    type=int,   default=50,   help='Max MPC steps')
    p.add_argument('--threshold',type=float, default=2.0,  help='Goal embedding dist threshold')
    p.add_argument('--settle',   type=float, default=0.5,  help='Settle time after move (s)')
    return p.parse_args()


def main():
    ACTION_SCALE = 0.01

    args = parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── load model ────────────────────────────────────────────────────────────
    vae, encoder, projector, delta_embedder, ldp = build_model(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        candidates = sorted(Path('checkpoints').glob('jepa_epoch_*.pt'))
        if not candidates:
            raise FileNotFoundError('No jepa_epoch_*.pt checkpoints found in checkpoints/.')
        ckpt_path = candidates[-1]

    load_checkpoint(ckpt_path, encoder, projector, delta_embedder, ldp, device)

    # ── load goal image ────────────────────────────────────────────────────────
    def read_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    goal_image = read_rgb(args.goal)

    # ── liveview subprocess ────────────────────────────────────────────────────
    liveview_proc = None
    cam           = None

    if not args.dry_run:
        liveview_script = Path(__file__).parent / 'liveview.py'
        liveview_proc   = subprocess.Popen(
            [sys.executable, str(liveview_script), '--index', '0', '--fps', '15'],
            stdin=subprocess.PIPE,
        )
        from hardware.camera_controller import CameraController
        cam = CameraController(index=0, fps=15)
        cam.start()

    # ── robot ─────────────────────────────────────────────────────────────────
    robot = None
    if not args.dry_run and not args.no_motion:
        from hardware.transfer_control_controller import TransferControl
        robot = TransferControl(only_xyz=True)
        print('Robot connected.')

    if args.no_motion:
        print('No-motion mode — camera live, moves skipped.')

    # ── run ───────────────────────────────────────────────────────────────────
    cumulative = {ax: 0.0 for ax in ('x', 'y', 'z')}
    try:
        cumulative = run_mpc(
            goal_image,
            vae, encoder, projector, delta_embedder, ldp,
            robot,
            cam,
            liveview_proc,
            device=device,
            max_steps=args.steps,
            goal_threshold=args.threshold,
            horizon=args.horizon,
            n_samples=args.samples,
            n_elite=args.elite,
            n_iters=args.iters,
            action_std=args.std,
            action_min=-args.max_step,
            action_max=args.max_step,
            action_scale=ACTION_SCALE,
            settle_s=args.settle,
        )
    except KeyboardInterrupt:
        print('\nInterrupted — homing robot to start position...')
        if robot is not None:
            for ax, total in cumulative.items():
                if total != 0.0:
                    print(f'  return {ax} by {-total:+.3f} mm')
                    robot.move_axis_by(ax, -total)
            print('Homing complete.')
    finally:
        if liveview_proc is not None:
            liveview_proc.terminate()
            liveview_proc.wait()
        if cam is not None:
            cam.stop()
        if robot is not None:
            robot.disconnect()


if __name__ == '__main__':
    main()
