"""
Calibrate px_per_mm: capture a frame, move the x axis by a known mm amount,
capture another frame, and measure the pixel displacement between them via
SIFT feature matching. px_per_mm = pixel_displacement / mm_displacement.

Usage:
    python calibrate_px_per_mm.py --dx 0.1
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
    p = argparse.ArgumentParser(description='Calibrate px_per_mm using SIFT feature matching.')
    p.add_argument('--dx', type=float, default=-0.1, help='Test move Δx (mm)')
    p.add_argument('--settle', type=float, default=0.5, help='Settle time after move (s)')
    p.add_argument('--ratio', type=float, default=0.75, help="Lowe's ratio test threshold")
    return p.parse_args()


def sift_displacement(img1: np.ndarray, img2: np.ndarray, ratio: float) -> np.ndarray:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        raise RuntimeError('Could not find SIFT features in one or both frames.')

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f'Only {len(good)} good matches found — cannot reliably estimate displacement.')

    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)

    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError('Affine estimation failed.')

    n_inliers = int(inliers.sum()) if inliers is not None else len(good)
    print(f'SIFT matches: {len(good)} good, {n_inliers} inliers')

    disp_px = M[:, 2]   # (dx_px, dy_px), image coords (y down)
    return disp_px


def main():
    args = parse_args()

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

        print(f'Moving x by {args.dx:+.3f} mm ...')
        robot.move_axis_by('x', args.dx, timeout_ms=10000)

        frame2 = cam.snap()
        pos2   = robot.positions()
        print(f'Position 2: {pos2}')

        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)
        cv2.waitKey(0)

        dx_mm = pos2['x'] - pos1['x']
        print(f'Encoder-reported Δx: {dx_mm:+.4f} mm')

    finally:
        cam.stop()
        robot.disconnect()

    disp_px = sift_displacement(frame1, frame2, args.ratio)
    disp_mag_px = float(np.hypot(disp_px[0], disp_px[1]))
    print(f'SIFT-measured displacement: ({disp_px[0]:+.2f}, {disp_px[1]:+.2f}) px  '
          f'(magnitude {disp_mag_px:.2f} px)')

    if abs(dx_mm) < 1e-9:
        raise RuntimeError('Encoder-reported Δx is ~0 — cannot compute px_per_mm.')

    px_per_mm = disp_mag_px / abs(dx_mm)
    print(f'\npx_per_mm = {px_per_mm:.4f}')


if __name__ == '__main__':
    main()
