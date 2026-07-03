"""
Validate the transfer stage's position encoder against what the camera actually sees.

Procedure:
    1. Capture frame_1 and read encoder position_1.
    2. Move the stage by (dx, dy) mm (non-blocking, debounced on IsMoving).
    3. Capture frame_2 and read encoder position_2.
    4. delta_mm = position_2 - position_1  (the encoder's account of what moved)
    5. Warp frame_1 by delta_mm * px_per_mm (x/y only) using cv2.warpAffine.
    6. Show [frame_1 | warped frame_1 | frame_2] side by side.

If the encoder + calibration (--px_per_mm) are correct, the warped frame_1
should visually line up with frame_2.

Usage:
    python validate_encoder.py --dx 1.0 --dy 0.0
    python validate_encoder.py --img1 a.png --img2 b.png --dx 1.0 --dy 0.0 --dry_run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description='Validate transfer stage position encoder against camera.')
    p.add_argument('--dx', type=float, default=1.0, help='Test move Δx (mm)')
    p.add_argument('--dy', type=float, default=0.0, help='Test move Δy (mm)')
    p.add_argument('--px_per_mm', type=float, default=8.0, help='Calibration: pixels per mm')
    p.add_argument('--settle', type=float, default=0.5, help='Settle time after move (s)')
    p.add_argument('--dry_run', action='store_true',
                   help='Use --img1/--img2 files instead of live camera/robot')
    p.add_argument('--img1', default=None, help='(dry_run) path to frame_1')
    p.add_argument('--img2', default=None, help='(dry_run) path to frame_2')
    return p.parse_args()


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()

    if args.dry_run:
        if not args.img1 or not args.img2:
            raise ValueError('--dry_run requires --img1 and --img2')
        frame1 = cv2.imread(args.img1)
        frame2 = cv2.imread(args.img2)
        if frame1 is None or frame2 is None:
            raise FileNotFoundError('Could not read --img1/--img2')
        delta_mm = np.array([args.dx, args.dy], dtype=np.float32)

    else:
        from hardware.camera_controller import CameraController
        from hardware.transfer_control_controller import TransferControl

        cam   = CameraController(index=0, fps=15)
        robot = TransferControl(only_xyz=True)

        try:
            cam.start()
            robot.connect()

            frame1 = cam.snap()
            pos1   = robot.positions()
            print(f'Position 1: {pos1}')

            print(f'Moving x by {args.dx:+.3f} mm, y by {args.dy:+.3f} mm ...')
            robot.move_axis_by('x', args.dx, timeout_ms=0)
            robot.move_axis_by('y', args.dy, timeout_ms=0)

            x_axis = robot._get_axis('x')
            y_axis = robot._get_axis('y')
            while x_axis.dev.Status.IsMoving or y_axis.dev.Status.IsMoving:
                time.sleep(0.02)
            time.sleep(args.settle)

            frame2 = cam.snap()
            pos2   = robot.positions()
            print(f'Position 2: {pos2}')

            delta_mm = np.array([pos2['x'] - pos1['x'], pos2['y'] - pos1['y']], dtype=np.float32)

        finally:
            cam.stop()
            robot.disconnect()

    print(f'Encoder-reported Δ(x, y): {delta_mm} mm')

    delta_px = delta_mm * args.px_per_mm   # (dx_px, dy_px)
    M = np.array([[1, 0, delta_px[0]],
                  [0, 1, -delta_px[1]]], dtype=np.float32)   # image y grows downward
    h, w = frame1.shape[:2]
    warped = cv2.warpAffine(frame1, M, (w, h))

    def to_bgr(img):
        return img if img.ndim == 3 and img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    frame1  = to_bgr(frame1)
    warped  = to_bgr(warped)
    frame2  = to_bgr(frame2)

    panel = np.hstack([
        label(frame1, 'frame_1'),
        label(warped, f'warped ({delta_mm[0]:+.3f}, {delta_mm[1]:+.3f}) mm'),
        label(frame2, 'frame_2'),
    ])

    cv2.namedWindow('Encoder validation', cv2.WINDOW_NORMAL)
    cv2.imshow('Encoder validation', panel)
    print('Press ESC or any key to close.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
