"""
Interactively overlay a resized still image on top of frames from a converted HDF5 recording, with
trackbars to scrub through frames and control overlay opacity, and WASD/arrow keys to nudge the
overlay's position.

`--img` (default 20X_DropDown/0/hBN_1.jpg) is resized to match the magnification of `--hdf5` (default
20X_DropDown/0/5x.hdf5) and composited onto its 224x224 frames: 20x recordings are 4x more magnified
than 5x ones, so the overlay is resized to 224x224 for a 20x file and 56x56 for a 5x file. The overlay
starts centered on the frame; nudging it updates the on-screen pixel displacement from that starting
position. Each frame's x/y/z mm position (stored in the HDF5 file itself) is shown alongside it.

Pressing 'f' toggles follow mode: the overlay position is locked to wherever it was when follow mode
was turned on, then driven purely by the change in mm position (via a fixed px/mm ratio) as you scrub
frames, instead of by manual nudging.

Controls:
    'frame' trackbar   — scrub to any frame
    'opacity' trackbar — overlay opacity, 0-100%
    Arrow keys / WASD  — nudge the overlay image by --step px per press
    f                  — toggle follow mode
    ESC                — quit

Usage:
    python video_overlay_align.py
    python video_overlay_align.py --img 20X_DropDown/0/hBN_1.jpg --hdf5 20X_DropDown/0/20x.hdf5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description='Interactively align a resized image on top of an HDF5 recording.')
    p.add_argument('--img', default=str(REPO_ROOT / '20X_DropDown' / '0' / 'hBN_1.jpg'), help='Path to the still image to resize and overlay')
    p.add_argument('--hdf5', default=str(REPO_ROOT / '20X_DropDown' / '0' / '5x.hdf5'), help='Path to a converted HDF5 recording (frame_i_x/frame_i_y datasets)')
    p.add_argument('--step', type=int, default=2, help='Pixels moved per key press when nudging the overlay')
    return p.parse_args()


def overlay_size_for(hdf5_path: Path) -> int:
    """20x recordings are 4x more magnified than 5x ones, so the (physical-space) overlay image
    needs to be shown 4x larger in pixels to line up: 224x224 for 20x, 56x56 for 5x."""
    name = hdf5_path.name.lower()
    if '20x' in name:
        return 224
    if '5x' in name:
        return 56
    raise ValueError(f"Can't tell magnification from filename: {hdf5_path.name!r} (expected '20x' or '5x' in the name)")


# (x, y) px-per-mm for follow mode, measured at 5x: +1mm in x is -72px, +1mm in y is +150px.
RATIO_5X_PX_PER_MM = np.array([-72.0, 150.0], dtype=np.float32)
# Not measured yet at 20x — assumed to scale with magnification, same as overlay_size_for.
RATIO_20X_PX_PER_MM = RATIO_5X_PX_PER_MM * 4.0


def ratio_for(hdf5_path: Path) -> np.ndarray:
    """px-per-mm conversion for follow mode, keyed off the same 20x/5x filename convention as
    overlay_size_for."""
    name = hdf5_path.name.lower()
    if '20x' in name:
        return RATIO_20X_PX_PER_MM
    if '5x' in name:
        return RATIO_5X_PX_PER_MM
    raise ValueError(f"Can't tell magnification from filename: {hdf5_path.name!r} (expected '20x' or '5x' in the name)")


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (5, 10 + idx * 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()

    img_full = cv2.imread(args.img)
    if img_full is None:
        raise FileNotFoundError(f'Could not read image: {args.img}')

    hdf5_path = Path(args.hdf5)
    overlay_size = overlay_size_for(hdf5_path)
    ratio = ratio_for(hdf5_path)
    small = cv2.resize(img_full, (overlay_size, overlay_size), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    with h5py.File(hdf5_path, 'r') as hf:
        n_frames = sum(1 for k in hf.keys() if k.endswith('_x'))
        vh, vw = hf['frame_0_x'].shape[:2]
        print(f'{hdf5_path}: {n_frames} frames  {vw}x{vh}  (overlay resized to {overlay_size}x{overlay_size})')

        # overlay starts centered on the frame — displacement is reported relative to this
        start_x, start_y = (vw - sw) // 2, (vh - sh) // 2
        off_x, off_y = start_x, start_y

        follow = False
        follow_ref_mm = None
        follow_ref_off = (off_x, off_y)

        window = 'Video Overlay Align'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)

        frame_idx = 0

        def on_frame(pos):
            nonlocal frame_idx
            frame_idx = pos

        cv2.createTrackbar('frame', window, 0, max(n_frames - 1, 1), on_frame)
        cv2.createTrackbar('opacity', window, 50, 100, lambda _pos: None)

        print('Use the trackbars to scrub/set opacity. Arrow keys / WASD nudge the overlay. f toggles follow mode. ESC quits.')
        while True:
            cv2.setTrackbarPos('frame', window, frame_idx)
            frame = hf[f'frame_{frame_idx}_x'][:]
            pos_mm = hf[f'frame_{frame_idx}_y'][:]

            if follow:
                delta_mm = pos_mm[:2] - follow_ref_mm
                off_x = int(round(follow_ref_off[0] + delta_mm[0] * ratio[0]))
                off_y = int(round(follow_ref_off[1] + delta_mm[1] * ratio[1]))

            opacity = cv2.getTrackbarPos('opacity', window) / 100.0

            M = np.array([[1, 0, off_x], [0, 1, off_y]], dtype=np.float32)
            warped = cv2.warpAffine(small, M, (vw, vh))
            overlay = cv2.addWeighted(warped, opacity, frame, 1.0 - opacity, 0.0)

            display = label(overlay, [
                f'frame {frame_idx}/{n_frames - 1}',
                f'x={pos_mm[0]:+.3f}  y={pos_mm[1]:+.3f}  z={pos_mm[2]:+.3f} mm',
                f'overlay displacement: ({off_x - start_x:+d}, {off_y - start_y:+d}) px',
                f"follow mode: {'ON' if follow else 'OFF'} (f)",
            ])
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:                                        # ESC
                break
            elif key == ord('f'):
                follow = not follow
                if follow:
                    follow_ref_mm = pos_mm[:2].copy()
                    follow_ref_off = (off_x, off_y)
            elif key in (ord('d'), 83):                          # right / d
                off_x += args.step
            elif key in (ord('a'), 81):                           # left / a
                off_x -= args.step
            elif key in (ord('s'), 84):                           # down / s
                off_y += args.step
            elif key in (ord('w'), 82):                           # up / w
                off_y -= args.step

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
