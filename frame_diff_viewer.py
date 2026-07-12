"""
View the raw pixel difference between two frames to check how visible the
film's displacement actually is to the naked eye (and to a network taking a
subtracted image as input).

Frames can come from either an HDF5 file (--h5/--idx1/--idx2) or standalone
image files (--img1/--img2, e.g. PNGs). If the image filenames follow the
`..._x_y_z.png` convention (as in images/capture_x_y_z.png), the ground-truth
Δ(x, y, z) is parsed from them and printed, same as for HDF5 frames.

Usage:
    python frame_diff_viewer.py --h5 robot_1.hdf5 --idx1 0 --idx2 5
    python frame_diff_viewer.py --h5 robot_1.hdf5 --idx1 0 --idx2 5 --data_dir data --amplify 4
    python frame_diff_viewer.py --img1 images/capture_6.08_4.48_5.12.png --img2 images/capture_6.31_4.78_5.05.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

FNAME_RE = re.compile(r'([-\d.eE]+)_([-\d.eE]+)_([-\d.eE]+)\.(?:png|jpg|jpeg)$', re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description='View the pixel difference between two frames (HDF5 frames or standalone images).')
    p.add_argument('--h5', help='HDF5 filename (e.g. robot_1.hdf5)')
    p.add_argument('--data_dir', default='data', help='Directory containing the HDF5 file')
    p.add_argument('--idx1', type=int, help='Frame index 1 (with --h5)')
    p.add_argument('--idx2', type=int, help='Frame index 2 (with --h5)')
    p.add_argument('--img1', help='Path to image 1 (alternative to --h5/--idx1)')
    p.add_argument('--img2', help='Path to image 2 (alternative to --h5/--idx2)')
    p.add_argument('--amplify', type=float, default=1.0, help='Multiply the (signed) diff before clipping, to make small shifts more visible')
    p.add_argument('--blur', type=float, default=0.0, help='Gaussian blur sigma applied to each frame before subtraction (0 = no blur)')
    p.add_argument('--threshold', type=float, default=0.0, help='Zero out diff values with |diff| below this threshold (0 = no thresholding)')
    args = p.parse_args()

    using_h5  = args.h5 is not None or args.idx1 is not None or args.idx2 is not None
    using_img = args.img1 is not None or args.img2 is not None
    if using_h5 and using_img:
        p.error('Use either --h5/--idx1/--idx2 or --img1/--img2, not both.')
    if using_h5 and (args.h5 is None or args.idx1 is None or args.idx2 is None):
        p.error('--h5 requires both --idx1 and --idx2.')
    if using_img and (args.img1 is None or args.img2 is None):
        p.error('--img1 and --img2 must both be given.')
    if not using_h5 and not using_img:
        p.error('Must supply either --h5/--idx1/--idx2 or --img1/--img2.')

    return args


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


def load_image(path: str):
    """Load a standalone image file. Parses (x, y, z) mm position from a
    `..._x_y_z.png` filename if present, else returns position=None."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    m = FNAME_RE.search(Path(path).name)
    position = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=np.float32) if m else None
    return bgr[:, :, ::-1], position   # BGR -> RGB


def main():
    args = parse_args()

    if args.img1 is not None:
        frame1, pos1 = load_image(args.img1)
        frame2, pos2 = load_image(args.img2)
        label1, label2 = Path(args.img1).name, Path(args.img2).name
    else:
        data_dir = Path(args.data_dir)
        frame1, pos1 = load_frame(data_dir, args.h5, args.idx1)
        frame2, pos2 = load_frame(data_dir, args.h5, args.idx2)
        label1, label2 = f'frame {args.idx1}', f'frame {args.idx2}'

    if pos1 is not None and pos2 is not None:
        delta_mm = pos2[:2] - pos1[:2]
        delta_z_mm = pos2[2] - pos1[2]
        print(f'Ground-truth Δ(x, y): {delta_mm} mm')
        print(f'Ground-truth Δz: {delta_z_mm:.4f} mm')
    else:
        delta_mm = None
        delta_z_mm = None
        print('No ground-truth position available for these frames.')

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
    axes[0].set_title(label1)
    axes[0].axis('off')

    axes[1].imshow(frame2)
    axes[1].set_title(label2)
    axes[1].axis('off')

    axes[2].imshow(display_diff_uint)
    axes[2].set_title(f'signed diff (RGB, amplify={args.amplify}x, blur σ={args.blur}, thresh={args.threshold})')
    axes[2].axis('off')

    gt_str = f'Δ(x,y)={delta_mm} mm   Δz={delta_z_mm:.4f} mm' if delta_mm is not None else 'no ground truth available'
    heat = axes[3].imshow(abs_diff_gray, cmap='inferno', vmin=0, vmax=abs_diff_gray.max())
    axes[3].set_title(f'|diff| grayscale heatmap\n{gt_str}')
    axes[3].axis('off')
    fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
