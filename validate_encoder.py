"""
Validate an affine-transform calibration by warping frame_1 to match frame_2.

No motors are driven here. You supply frame_1, frame_2, and the known/assumed
(dx, dy) displacement (mm) between them (e.g. moved the stage by hand, or read
it off the encoder yourself). The script warps frame_1 by dx,dy * px_per_mm and
shows [frame_1 | warped frame_1 | frame_2] side by side so you can visually
confirm the transform lines up.

Usage:
    python validate_encoder.py --img1 a.png --img2 b.png --dx 1.0 --dy 0.0
    python validate_encoder.py --img1 a.png --img2 b.png --dx 1.0 --dy 0.0 --px_per_mm 8.0
    python validate_encoder.py --dx 1.0 --dy 0.0   # capture frame_1/frame_2 live from camera only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description='Validate transfer stage position encoder against camera.')
    p.add_argument('--dx', type=float, default=1.0, help='Test move Δx (mm)')
    p.add_argument('--dy', type=float, default=0.0, help='Test move Δy (mm)')
    p.add_argument('--px_per_mm', type=float, default=8.0, help='Calibration: pixels per mm')
    p.add_argument('--img1', default=None, help='Path to frame_1 (omit to capture live from camera)')
    p.add_argument('--img2', default=None, help='Path to frame_2 (omit to capture live from camera)')
    return p.parse_args()


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()

    if args.img1 and args.img2:
        frame1 = cv2.imread(args.img1)
        frame2 = cv2.imread(args.img2)
        if frame1 is None or frame2 is None:
            raise FileNotFoundError('Could not read --img1/--img2')

    else:
        from hardware.camera_controller import CameraController

        cam = CameraController(index=0, fps=15)
        try:
            cam.start()
            input('Position 1 ready. Press Enter to capture frame_1...')
            frame1 = cam.snap()
            input('Now move the stage by hand (or however you like), then press Enter to capture frame_2...')
            frame2 = cam.snap()
        finally:
            cam.stop()

    delta_mm = np.array([args.dx, args.dy], dtype=np.float32)
    print(f'Assumed Δ(x, y): {delta_mm} mm')

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
