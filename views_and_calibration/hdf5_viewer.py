"""
Scrub through a robot_*.hdf5 recording frame by frame (or play it back), with a stats overlay
(frame index, x/y position in mm, ring area) and a green marker on the Newton-ring centroid
(cx, cy) when one was found for that frame.

Controls:
    Trackbar           — scrub to any frame
    Space              — play / pause
    Left / Right / a / d — step one frame back / forward (pauses playback)
    ESC                — quit

Usage:
    python hdf5_viewer.py                       # pick a file interactively from --data_dir
    python hdf5_viewer.py --file robot_3.hdf5
    python hdf5_viewer.py --file robot_3.hdf5 --fps 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description='Scrub/play back a robot_*.hdf5 recording with stats + ring overlay.')
    p.add_argument('--data_dir', default=str(REPO_ROOT / 'data'), help='Directory containing robot_*.hdf5 files')
    p.add_argument('--file', default=None, help='HDF5 filename to open (e.g. robot_3.hdf5). If omitted, pick interactively.')
    p.add_argument('--fps', type=float, default=30.0, help='Playback speed in frames per second')
    return p.parse_args()


def pick_file(data_dir: Path) -> Path:
    files = sorted(data_dir.glob('robot_*.hdf5'))
    if not files:
        raise FileNotFoundError(f'No robot_*.hdf5 files found in {data_dir}')
    print('Available HDF5 files:')
    for idx, path in enumerate(files):
        print(f'  [{idx}] {path.name}')
    choice = input(f'Pick a file [0-{len(files) - 1}]: ').strip()
    return files[int(choice)]


def num_frames(f: h5py.File) -> int:
    return sum(1 for k in f.keys() if k.endswith('_x'))


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (10, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def render_frame(f: h5py.File, idx: int, n: int, scale: int = 3) -> np.ndarray:
    frame_bgr = f[f'frame_{idx}_x'][:].copy()
    frame_bgr = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pos = f[f'frame_{idx}_y'][:].astype(np.float32)
    ring_key = f'frame_{idx}_ring'
    ring = f[ring_key][:] if ring_key in f else np.full(3, np.nan, dtype=np.float32)

    lines = [
        f'frame {idx}/{n - 1}',
        f'x={pos[0]:+.3f} mm  y={pos[1]:+.3f} mm',
    ]
    if not np.isnan(ring).any():
        cx, cy, area = ring
        cx *= scale
        cy *= scale
        lines.append(f'ring area={area:.1f} px^2')
        cv2.circle(frame_bgr, (int(round(cx)), int(round(cy))), 6, (0, 255, 0), -1)
    else:
        lines.append('ring area=n/a')

    return label(frame_bgr, lines)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    path = Path(args.file) if args.file is not None else pick_file(data_dir)
    if not path.is_absolute() and not path.exists():
        path = data_dir / path

    with h5py.File(path, 'r') as f:
        n = num_frames(f)
        print(f'{path.name}: {n} frames')

        window = 'HDF5 Viewer'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 540)

        idx = 0
        playing = False
        delay_ms = max(1, int(round(1000 / args.fps)))

        def on_trackbar(pos):
            nonlocal idx
            idx = pos

        cv2.createTrackbar('frame', window, 0, n - 1, on_trackbar)

        print('Space = play/pause, Left/Right or a/d = step, ESC = quit.')
        while True:
            cv2.setTrackbarPos('frame', window, idx)
            cv2.imshow(window, render_frame(f, idx, n))

            key = cv2.waitKey(delay_ms if playing else 30) & 0xFF
            if key == 27:                                    # ESC
                break
            elif key == ord(' '):
                playing = not playing
            elif key in (ord('d'), 83):                       # right / d
                playing = False
                idx = min(idx + 1, n - 1)
            elif key in (ord('a'), 81):                       # left / a
                playing = False
                idx = max(idx - 1, 0)
            elif playing:
                idx = idx + 1
                if idx >= n:
                    idx = n - 1
                    playing = False

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
