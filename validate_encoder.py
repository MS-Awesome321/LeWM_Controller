"""
Validate the displacement model by warping frame_1 to match frame_2.

Loads the Siamese-ViT displacement model (checkpoints/disp_epoch_*.pt), runs
it on (img1, img2) to predict (Δx, Δy, Δz) in mm, then warps img1 by
(Δx, Δy) * px_per_mm using cv2.warpAffine and shows
[frame_1 | warped frame_1 | frame_2] side by side so you can visually confirm
the prediction lines up.

Usage:
    python validate_encoder.py --img1 a.png --img2 b.png
    python validate_encoder.py --img1 a.png --img2 b.png --ckpt checkpoints/disp_epoch_0080.pt --px_per_mm 8.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from transformers import ViTModel, ViTConfig

IMG_SIZE = 224
EMB_DIM  = 192


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


def load_checkpoint(ckpt_path: Path, encoder, head, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    delta_mean = ckpt['delta_mean'].to(device)
    delta_std  = ckpt['delta_std'].to(device)
    print(f'Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get("epoch", "?")})')
    return delta_mean, delta_std


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


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Validate the displacement model against two images.')
    p.add_argument('--img1', required=True, help='Path to frame_1')
    p.add_argument('--img2', required=True, help='Path to frame_2')
    p.add_argument('--ckpt', default='checkpoints/disp_epoch_0080.pt', help='Path to displacement checkpoint (.pt)')
    p.add_argument('--px_per_mm', type=float, default=8.0, help='Calibration: pixels per mm')
    return p.parse_args()


def main():
    args = parse_args()

    frame1 = cv2.imread(args.img1)
    frame2 = cv2.imread(args.img2)
    if frame1 is None or frame2 is None:
        raise FileNotFoundError('Could not read --img1/--img2')

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    encoder, head = build_model(device)
    delta_mean, delta_std = load_checkpoint(Path(args.ckpt), encoder, head, device)

    emb1 = encode_frame(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), encoder, device)
    emb2 = encode_frame(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), encoder, device)
    pred_xyz = predict_displacement(emb1, emb2, head, delta_mean, delta_std)
    delta_mm = pred_xyz[:2]
    print(f'Predicted Δ(x, y, z): {pred_xyz} mm')

    delta_px = delta_mm * args.px_per_mm   # (dx_px, dy_px)
    M = np.array([[1, 0, delta_px[0]],
                  [0, 1, -delta_px[1]]], dtype=np.float32)   # image y grows downward
    h, w = frame1.shape[:2]
    warped = cv2.warpAffine(frame1, M, (w, h))

    panel = np.hstack([
        label(frame1, 'frame_1'),
        label(warped, f'warped ({delta_mm[0]:+.3f}, {delta_mm[1]:+.3f}) mm'),
        label(frame2, 'frame_2'),
    ])

    cv2.namedWindow('Displacement validation', cv2.WINDOW_NORMAL)
    cv2.imshow('Displacement validation', panel)
    print('Press ESC or any key to close.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
