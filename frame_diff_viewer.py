"""
View the raw pixel difference between two HDF5 frames to check how visible the
film's displacement actually is to the naked eye (and to a network taking a
subtracted image as input).

Usage:
    python frame_diff_viewer.py --h5 robot_1.hdf5 --idx1 0 --idx2 5
    python frame_diff_viewer.py --h5 robot_1.hdf5 --idx1 0 --idx2 5 --data_dir data --amplify 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='View the pixel difference between two HDF5 frames.')
    p.add_argument('--h5', required=True, help='HDF5 filename (e.g. robot_1.hdf5)')
    p.add_argument('--data_dir', default='data', help='Directory containing the HDF5 file')
    p.add_argument('--idx1', type=int, required=True, help='Frame index 1')
    p.add_argument('--idx2', type=int, required=True, help='Frame index 2')
    p.add_argument('--amplify', type=float, default=1.0, help='Multiply the (signed) diff before clipping, to make small shifts more visible')
    p.add_argument('--blur', type=float, default=0.0, help='Gaussian blur sigma applied to each frame before subtraction (0 = no blur)')
    p.add_argument('--threshold', type=float, default=0.0, help='Zero out diff values with |diff| below this threshold (0 = no thresholding)')
    return p.parse_args()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def load_frame(data_dir: Path, h5_name: str, idx: int):
    path = data_dir / h5_name
    with h5py.File(path, 'r') as f:
        frame_bgr = f[f'frame_{idx}_x'][:]
        position = f[f'frame_{idx}_y'][:].astype(np.float32)
    return frame_bgr[:, :, ::-1], position   # BGR -> RGB


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    frame1, pos1 = load_frame(data_dir, args.h5, args.idx1)
    frame2, pos2 = load_frame(data_dir, args.h5, args.idx2)
    delta_mm = pos2[:2] - pos1[:2]
    delta_z_mm = pos2[2] - pos1[2]
    print(f'Ground-truth Δ(x, y): {delta_mm} mm')
    print(f'Ground-truth Δz: {delta_z_mm:.4f} mm')

    f1 = gaussian_blur(frame1.astype(np.float32), args.blur)
    f2 = gaussian_blur(frame2.astype(np.float32), args.blur)

    signed_diff = f2 - f1                          # (H, W, 3), range [-255, 255]
    if args.threshold > 0:
        signed_diff = np.where(np.abs(signed_diff) < args.threshold, 0.0, signed_diff)
    abs_diff_gray = np.abs(signed_diff).mean(axis=-1)   # (H, W), range [0, 255]

    display_diff = np.clip(signed_diff * args.amplify, -255, 255)
    display_diff_uint = ((display_diff + 255) / 2).astype(np.uint8)   # remap to [0,255] for imshow

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(frame1)
    axes[0].set_title(f'frame {args.idx1}')
    axes[0].axis('off')

    axes[1].imshow(frame2)
    axes[1].set_title(f'frame {args.idx2}')
    axes[1].axis('off')

    axes[2].imshow(display_diff_uint)
    axes[2].set_title(f'signed diff (RGB, amplify={args.amplify}x, blur σ={args.blur}, thresh={args.threshold})')
    axes[2].axis('off')

    heat = axes[3].imshow(abs_diff_gray, cmap='inferno', vmin=0, vmax=abs_diff_gray.max())
    axes[3].set_title(f'|diff| grayscale heatmap\nΔ(x,y)={delta_mm} mm   Δz={delta_z_mm:.4f} mm')
    axes[3].axis('off')
    fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
