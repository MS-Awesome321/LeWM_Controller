"""
Interactive patch-similarity explorer for the diff-pred ViT (trained in diff_pred.ipynb).

Loads two frames (by index) out of a robot_*.hdf5 file, computes the same blurred difference
image the model was trained on (blur(frame2) - blur(frame1)), runs it through the trained ViT
encoder to get per-patch features, and shows the diff as both an RGB image and a magnitude
heatmap. Hover over either to see **Euclidean** (not cosine) distance-based similarity of that
patch against every patch in the diff, rendered as a heatmap overlay.

Usage:
    python vit_feat.py --h5 robot_1.hdf5 --idx1 0 --idx2 500
    python vit_feat.py --h5 robot_1.hdf5 --idx1 0 --idx2 500 --data_dir data
    python vit_feat.py --h5 robot_1.hdf5 --idx1 0 --idx2 500 --ckpt checkpoints/diff_pred_epoch_0040.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

IMG_SIZE    = 224
PATCH_SIZE  = 16
EMB_DIM     = 192
BLUR_SIGMA  = 2.0   # must match training
AXIS_NAMES  = ['x', 'y', 'cx', 'cy', 'area']
AXIS_UNITS  = ['mm', 'mm', 'px', 'px', 'px²']


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

    def forward(self, emb):
        return self.net(emb)


def parse_args():
    p = argparse.ArgumentParser(description='Explore the diff-pred ViT patch similarity (Euclidean) between two HDF5 frames.')
    p.add_argument('--h5', required=True, help='HDF5 filename (e.g. robot_1.hdf5).')
    p.add_argument('--idx1', type=int, required=True, help='Frame index into --h5.')
    p.add_argument('--idx2', type=int, required=True, help='Frame index into --h5.')
    p.add_argument('--data_dir', default='data', help='Directory containing the HDF5 file')
    p.add_argument('--ckpt', default='checkpoints/diff_pred_epoch_0040.pt', help='diff_pred.ipynb checkpoint to load the ViT encoder from')
    return p.parse_args()


def load_frame(data_dir: Path, h5_name: str, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = data_dir / h5_name
    with h5py.File(path, 'r') as f:
        frame_bgr = f[f'frame_{idx}_x'][:]
        position  = f[f'frame_{idx}_y'][:]
        ring_key  = f'frame_{idx}_ring'
        ring      = f[ring_key][:] if ring_key in f else np.full(3, np.nan, dtype=np.float32)
    return frame_bgr[:, :, ::-1].copy(), position, ring   # BGR -> RGB


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


def to_float(rgb_u8: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1] on device."""
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t / 255.0


@torch.no_grad()
def extract_patch_features(diff: torch.Tensor, vit: ViTModel):
    """diff: (1, 3, IMG_SIZE, IMG_SIZE) -> (patch_h*patch_w, EMB_DIM) NOT L2-normalised
    (we compare patches with Euclidean distance, not cosine similarity, so normalising would
    throw away magnitude information)."""
    hidden = vit(diff, interpolate_pos_encoding=False).last_hidden_state   # (1, 1+N, EMB_DIM)
    patch_tokens = hidden[0, 1:]                                            # (N, EMB_DIM), drop CLS
    grid = IMG_SIZE // PATCH_SIZE
    return patch_tokens.cpu().numpy(), grid, grid


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    frame1, pos1, ring1 = load_frame(data_dir, args.h5, args.idx1)
    frame2, pos2, ring2 = load_frame(data_dir, args.h5, args.idx2)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    print(f'Loading ViT encoder from {args.ckpt} ...')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    vit_cfg = ViTConfig(
        num_channels=3, image_size=IMG_SIZE, patch_size=PATCH_SIZE,
        hidden_size=EMB_DIM, num_hidden_layers=6,
        num_attention_heads=3, intermediate_size=768,
    )
    vit = ViTModel(vit_cfg, add_pooling_layer=False).to(device).eval()
    vit.load_state_dict(ckpt['vit'])

    head = DisplacementHead(emb_dim=EMB_DIM, target_dim=5).to(device).eval()
    head.load_state_dict(ckpt['head'])
    delta_mean = ckpt['delta_mean'].to(device)
    delta_std  = ckpt['delta_std'].to(device)

    blur_kernel = make_gaussian_kernel(BLUR_SIGMA, device)

    f1 = to_float(frame1, device)
    f2 = to_float(frame2, device)
    with torch.no_grad():
        diff = gaussian_blur(f2, blur_kernel) - gaussian_blur(f1, blur_kernel)   # (1, 3, H, W), ~[-1, 1]

    diff_np = diff[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
    diff_mag = np.linalg.norm(diff_np, axis=-1)                # (H, W)

    with torch.no_grad():
        hidden = vit(diff, interpolate_pos_encoding=False).last_hidden_state   # (1, 1+N, EMB_DIM)
        cls_token = hidden[:, 0]                                                 # (1, EMB_DIM)
        pred_norm = head(cls_token)                                         # (1, 5)
        pred_raw  = (pred_norm * delta_std + delta_mean)[0].cpu().numpy()    # (5,)

    feat = hidden[0, 1:].cpu().numpy()   # (N, EMB_DIM), drop CLS
    grid = IMG_SIZE // PATCH_SIZE
    hf, wf = grid, grid
    print(f'Patch grid: {hf}x{wf}  ({feat.shape[0]} patches, dim={feat.shape[1]})')

    ring1_found = not np.isnan(ring1).any()
    ring2_found = not np.isnan(ring2).any()
    area1 = 0.0 if not ring1_found else ring1[2]
    area2 = 0.0 if not ring2_found else ring2[2]
    gt_xy   = pos2[:2] - pos1[:2]
    gt_area = area2 - area1
    if ring1_found and ring2_found:
        gt_cxcy = ring2[:2] - ring1[:2]
    else:
        gt_cxcy = np.array([np.nan, np.nan])
    gt = np.array([gt_xy[0], gt_xy[1], gt_cxcy[0], gt_cxcy[1], gt_area])

    lines = ['ground truth vs. predicted Δ:']
    for i, (name, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        gt_str = f'{gt[i]:+.2f}' if not np.isnan(gt[i]) else 'n/a'
        lines.append(f'  Δ{name}: gt={gt_str} {unit}   pred={pred_raw[i]:+.2f} {unit}')
    info_text = '\n'.join(lines)
    print(info_text)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.5))
    ax1.imshow(frame1)
    ax1.set_title(f'frame [{args.idx1}]')
    ax1.axis('off')

    ax2.imshow(frame2)
    ax2.set_title(f'frame [{args.idx2}]')
    ax2.axis('off')

    ax3.imshow(diff_mag, cmap='inferno')
    ax3.set_title('diff magnitude (||Δrgb||)')
    ax3.axis('off')

    H, W = diff_np.shape[:2]

    heat3 = ax3.imshow(np.zeros((hf, wf)), extent=(0, W, H, 0), cmap='viridis', alpha=0.55)

    fig.suptitle('Hover a patch to see Euclidean-distance similarity (brighter = closer, i.e. more similar). Press Enter to clear.')
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', family='monospace', fontsize=9)

    def patch_index(event, ax):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return None
        px = int(np.clip(event.xdata / W * wf, 0, wf - 1))
        py = int(np.clip(event.ydata / H * hf, 0, hf - 1))
        return py * wf + px

    def on_move(event):
        idx = patch_index(event, ax1)
        if idx is None:
            idx = patch_index(event, ax2)
        if idx is None:
            idx = patch_index(event, ax3)
        if idx is None:
            return

        source = feat[idx]                                    # (EMB_DIM,)
        dists = np.linalg.norm(feat - source, axis=1)          # (N,) Euclidean distance
        sim = -dists.reshape(hf, wf)                            # closer (dist=0) -> highest value

        heat3.set_data(sim)
        heat3.set_clim(sim.min(), sim.max())
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key != 'enter':
            return
        heat3.set_data(np.zeros((hf, wf)))
        heat3.set_clim(0, 1)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout(rect=(0, 0.14, 1, 1))
    plt.show()


if __name__ == '__main__':
    main()
