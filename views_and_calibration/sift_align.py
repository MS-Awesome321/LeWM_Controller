"""
Align a downscaled image onto a full-size one using SIFT correspondences + an affine transform.

`--small` (default 20x_design/hBN_2.jpg) is downsized by `--downscale` (default 4x), then matched
against `--large` (default 20x_design/5x.jpg, left at native resolution) via SIFT + a RANSAC-fit
affine transform (estimateAffine2D). Shows the SIFT correspondences in one cv2 window and the aligned
small image overlaid on the large one (opacity `--opacity`) in another.

Usage:
    python sift_align.py
    python sift_align.py --small 20x_design/hBN_2.jpg --large 20x_design/5x.jpg --downscale 4
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Align a downscaled image onto a full-size one via SIFT + affine transform.')
    p.add_argument('--small', default='20x_design/hBN_2.jpg', help='Path to the image to downsize and align')
    p.add_argument('--large', default='20x_design/5x.jpg', help='Path to the reference image (left at native resolution)')
    p.add_argument('--downscale', type=float, default=4.0, help='Factor by which --small is downsized before matching')
    p.add_argument('--opacity', type=float, default=0.5, help='Opacity of the aligned small image in the overlay')
    p.add_argument('--ratio', type=float, default=0.75, help="Lowe's ratio test threshold for match filtering")
    return p.parse_args()


def main():
    args = parse_args()

    small_full = cv2.imread(args.small)
    large = cv2.imread(args.large)
    if small_full is None:
        raise FileNotFoundError(f'Could not read image: {args.small}')
    if large is None:
        raise FileNotFoundError(f'Could not read image: {args.large}')

    small = cv2.resize(small_full, None, fx=1.0 / args.downscale, fy=1.0 / args.downscale, interpolation=cv2.INTER_AREA)

    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    large_gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(small_gray, None)
    kp2, des2 = sift.detectAndCompute(large_gray, None)
    print(f'Keypoints: small={len(kp1)}  large={len(kp2)}')

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if m.distance < args.ratio * n.distance]
    print(f'Good matches (ratio test): {len(good)}')
    if len(good) < 3:
        raise RuntimeError(f'Only {len(good)} good matches — need at least 3 to fit an affine transform.')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError('Affine transform estimation failed.')
    n_inliers = int(inlier_mask.sum())
    print(f'Affine inliers: {n_inliers}/{len(good)}')
    print(f'Affine matrix:\n{M}')

    match_img = cv2.drawMatches(small, kp1, large, kp2, good, None,
                                 matchesMask=inlier_mask.ravel().tolist(),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    h, w = large.shape[:2]
    warped = cv2.warpAffine(small, M, (w, h))
    overlay = cv2.addWeighted(warped, args.opacity, large, 1.0 - args.opacity, 0.0)

    cv2.namedWindow('SIFT Matches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SIFT Matches', 1600, 500)
    cv2.imshow('SIFT Matches', match_img)

    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Overlay', 960, 540)
    cv2.imshow('Overlay', overlay)

    print('Press any key (in an image window) to quit.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
