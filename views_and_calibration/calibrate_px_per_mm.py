"""
Calibrate px_per_mm interactively: pick two random frames (+ their true xyz
mm positions) from a random robot_*.hdf5 file in ./data, overlay one frame on
the other, and let you nudge the top frame around with WASD/arrow keys until
it lines up with the bottom frame. The implied px_per_mm is displayed live.

Controls:
    Arrow keys / WASD  — nudge the top (frame_i) image by 1 px per press
    ESC                — quit

Usage:
    python calibrate_px_per_mm.py
    python calibrate_px_per_mm.py --data_dir data --seed 0
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description='Interactively calibrate px_per_mm using HDF5 frame pairs.')
    p.add_argument('--data_dir', default=str(REPO_ROOT / 'data'), help='Directory containing robot_*.hdf5 files')
    p.add_argument('--seed', type=int, default=None, help='Random seed for file/frame selection')
    p.add_argument('--min_sep', type=int, default=50, help='Minimum index separation between the two frames')
    return p.parse_args()


def pick_frame_pair(data_dir: Path, min_sep: int, min_disp_mm: float = 0.05, max_tries: int = 200):
    files = sorted(data_dir.glob('robot_*.hdf5'))
    if not files:
        raise FileNotFoundError(f'No robot_*.hdf5 files found in {data_dir}')

    for _ in range(max_tries):
        path = random.choice(files)

        with h5py.File(path, 'r') as f:
            n = len([k for k in f.keys() if k.endswith('_y')])
            if n <= min_sep:
                continue
            i = random.randint(0, n - min_sep - 1)
            j = random.randint(i + min_sep, n - 1)

            pos_i = f[f'frame_{i}_y'][:].astype(np.float32)
            pos_j = f[f'frame_{j}_y'][:].astype(np.float32)

            if float(np.hypot(*(pos_j[:2] - pos_i[:2]))) < min_disp_mm:
                continue   # stage didn't actually move (or barely) between these frames — retry

            frame_i = f[f'frame_{i}_x'][:]
            frame_j = f[f'frame_{j}_x'][:]

        print(f'File: {path.name}   frame_i={i}  frame_j={j}')
        print(f'pos_i={pos_i}  pos_j={pos_j}')
        return frame_i, frame_j, pos_i, pos_j

    raise RuntimeError(f'Could not find a frame pair with >= {min_disp_mm} mm displacement after {max_tries} tries.')


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (10, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    data_dir = Path(args.data_dir)
    frame_i, frame_j, pos_i, pos_j = pick_frame_pair(data_dir, args.min_sep)

    delta_mm = pos_j[:2] - pos_i[:2]   # (dx, dy) mm, ground truth from encoder
    print(f'Ground-truth Δ(x, y): {delta_mm} mm')

    h, w = frame_i.shape[:2]

    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)

    # offset in pixels applied to frame_i (top image) to align it onto frame_j
    off_x, off_y = 0, 0

    print('Use arrow keys / WASD to nudge frame_i onto frame_j. Enter = new pair. ESC = quit.')
    while True:
        M = np.array([[1, 0, off_x],
                      [0, 1, off_y]], dtype=np.float32)
        warped = cv2.warpAffine(frame_i, M, (w, h))
        overlay = cv2.addWeighted(warped, 0.5, frame_j, 0.5, 0.0)

        # x and y are calibrated independently: the source frames were resized
        # from 3840x2160 down to a 224x224 square (non-uniform squish, since
        # the original aspect ratio is 16:9, not 1:1), so pixels/mm differs
        # per axis and a single isotropic px_per_mm would be wrong.
        px_per_mm_x = off_x / delta_mm[0] if abs(delta_mm[0]) > 1e-9 else float('nan')
        px_per_mm_y = off_y / delta_mm[1] if abs(delta_mm[1]) > 1e-9 else float('nan')

        display = label(overlay, [
            f'offset=({off_x:+d}, {off_y:+d}) px',
            f'px_per_mm_x={px_per_mm_x:.2f}',
            f'px_per_mm_y={px_per_mm_y:.2f}',
        ])
        cv2.imshow('Overlay', display)

        key = cv2.waitKeyEx(0)
        key_masked = key & 0xFF
        if key_masked == 27:                                        # ESC
            break
        elif key_masked in (13, 10):                                 # Enter
            frame_i, frame_j, pos_i, pos_j = pick_frame_pair(data_dir, args.min_sep)
            delta_mm = pos_j[:2] - pos_i[:2]
            print(f'Ground-truth Δ(x, y): {delta_mm} mm')
            off_x, off_y = 0, 0
        elif key in (81, 63234, 2424832) or key_masked == ord('a'):  # left / a
            off_x -= 1
        elif key in (83, 63235, 2555904) or key_masked == ord('d'):  # right / d
            off_x += 1
        elif key in (82, 63232, 2490368) or key_masked == ord('w'):  # up / w
            off_y -= 1
        elif key in (84, 63233, 2621440) or key_masked == ord('s'):  # down / s
            off_y += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
