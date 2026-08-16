"""
Flake designer: three linked windows for manually eyeballing a flake design, automatically
registering it onto its captured video via SIFT, and inspecting the residual difference.

Window 1 ('Manual Overlay') lets you eyeball-align --top (default 20X_DropDown/0/top.jpg) on top of
--bottom (default 20X_DropDown/0/bottom.jpg): an opacity trackbar blends them, and WASD / arrow keys
nudge --top's position (same convention as video_overlay_align.py).

Window 2 ('SIFT Video Align') reuses the SIFT + RANSAC-affine registration from sift_align.py (SIFT
features are computed on a Gaussian-blurred copy of --bottom, used only for that fit — everything
displayed still uses the unblurred image) to register --bottom (downsized by --downscale, since it's a
20x-scale design being placed onto a 5x-scale video — skipped for a 20x --video, which is already at
the design's magnification) onto frame
--frame-num of --video (that fixed transform is then reused to overlay --bottom/--top onto whichever
frame you scrub to with the 'frame' trackbar). --top is placed relative to --bottom using the manual
displacement set in Window 1 (the only thing tying Window 1 to Windows 2/3); its own opacity here is
independent of Window 1's. 'top opacity', 'bottom opacity', and 'video opacity' trackbars (all in
Window 2) are normalized (each divided by the sum of all three) into weights that sum to 1, then
blended as a straight weighted sum of the top/bottom/video layers.

Window 4 ('Subtractor') has its own 'frame' trackbar, independent of Window 2's, letting you pick a
second video frame (e.g. an empty-background reference) to subtract out.

Window 3 ('Diff Magnitude') shows |overlay - subtractor_frame| (Window 2's composited overlay — top +
bottom warped onto its current video frame — minus Window 4's frame), as a color heatmap.

Pass --model-resize / --mr to resize every displayed image (both manual-overlay stills and the views
derived from --video) to 224x224, matching the HDF5 conversion's model input size. Without it, all
displayed images are instead downsampled by a factor of 3 in each dimension (native resolution is
generally too large to render live at trackbar-scrubbing speed).

Controls:
    'opacity' trackbar (Window 1)                  — blend of --top over --bottom in Window 1 only
    WASD / arrow keys                              — nudge --top in Window 1 (also repositions it in Window 2)
    'frame' trackbar (Window 2)                    — scrub through --video
    'top' / 'bottom' / 'video opacity' (Window 2)   — normalized (sum-to-1) blend weights for each layer in Window 2
    1 / 2 / 3                                       — select top / bottom / video as the target of +/- blur control
    + / -                                            — increase/decrease the selected layer's Gaussian blur sigma by 0.05 (Window 2's composite/diff)
    'frame' trackbar (Window 4, Subtractor)         — pick the frame subtracted in Window 3
    --frame-num                                     — video frame index used for the SIFT fit (set once at startup, not a trackbar)
    ESC (any window)                                — quit

Usage:
    python flake_designer.py
    python flake_designer.py --top 20X_DropDown/2/top.jpg --bottom 20X_DropDown/2/bottom.jpg
    python flake_designer.py --video 20X_DropDown/2/5x.mp4 --mr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description='Manually align two images, then SIFT-register a design onto its video and inspect the residual difference.')
    p.add_argument('--top', default=str(REPO_ROOT / '20X_DropDown' / '0' / 'top.jpg'), help='Top image in the manual overlay')
    p.add_argument('--bottom', default=str(REPO_ROOT / '20X_DropDown' / '0' / 'bottom.jpg'), help='Base image in the manual overlay; also the "design" registered onto --video')
    p.add_argument('--video', default=None, help="Video to register --bottom onto (default: '5x.mp4' next to --bottom)")
    p.add_argument('--downscale', type=float, default=4.0, help='Factor --bottom is downsized by before SIFT-registering onto --video (20x design -> 5x video); ignored (treated as 1) if --video is a 20x video')
    p.add_argument('--step', type=int, default=2, help='Pixels moved per key press when nudging --top in the manual overlay')
    p.add_argument('--ratio', type=float, default=0.75, help="Lowe's ratio test threshold for SIFT match filtering")
    p.add_argument('--frame-num', type=int, default=0, dest='frame_num', help='Video frame index used as the SIFT reference frame for registering --bottom onto --video')
    p.add_argument('--model-resize', '--mr', dest='model_resize', action='store_true',
                    help='Resize every displayed image to 224x224 (matching the HDF5 model input size); '
                         'otherwise every displayed image is downsampled by a factor of 3 instead')
    return p.parse_args()


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (5, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def upscale_if_10x(img: np.ndarray, path: str) -> np.ndarray:
    """10x-scale stills are half the magnification of a 20x video/design, so upscale them 2x to match."""
    if '10x' in Path(path).name.lower():
        return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return img


def blur_img(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur img by sigma (no-op if sigma <= 0) — used for the per-layer top/bottom/video
    blur control in Window 2 (keys 1/2/3 select the layer, +/- adjust its sigma)."""
    if sigma <= 0:
        return img
    k = 2 * max(1, int(round(3 * sigma))) + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def sift_affine(small_gray: np.ndarray, large_gray: np.ndarray, ratio_thresh: float) -> np.ndarray:
    """SIFT + ratio-test + RANSAC-affine pipeline (same as sift_align.py)."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(small_gray, None)
    kp2, des2 = sift.detectAndCompute(large_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    if len(good) < 3:
        raise RuntimeError(f'Only {len(good)} good matches — need at least 3 to fit a transform.')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError('Affine transform estimation failed.')
    n_inliers = int(inlier_mask.sum())
    return M


def main():
    MAX_CLIP = 30
    args = parse_args()

    top_full = cv2.imread(args.top)
    bottom_full = cv2.imread(args.bottom)
    if top_full is None:
        raise FileNotFoundError(f'Could not read image: {args.top}')
    if bottom_full is None:
        raise FileNotFoundError(f'Could not read image: {args.bottom}')
    top_full = upscale_if_10x(top_full, args.top)
    bottom_full = upscale_if_10x(bottom_full, args.bottom)

    video_path = Path(args.video) if args.video else Path(args.bottom).parent / '5x.mp4'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Window 2 setup: SIFT-register bottom (downsized) onto one reference video frame --------
    # --downscale only applies to a 5x video: a 20x video is already at the design's magnification.
    downscale = 1.0 if '20x' in video_path.name.lower() else args.downscale
    bottom_small = cv2.resize(bottom_full, None, fx=1.0 / downscale, fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
    top_small = cv2.resize(top_full, None, fx=1.0 / downscale, fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_num)
    ok, ref_frame = cap.read()
    if not ok:
        raise RuntimeError(f'Could not read frame {args.frame_num} (SIFT reference) from {video_path}')
    bottom_small_blurred = cv2.GaussianBlur(bottom_small, (15, 15), 0)   # for SIFT features only — not used for warping/display
    M = sift_affine(cv2.cvtColor(bottom_small_blurred, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), args.ratio)
    warped_bottom = cv2.warpAffine(bottom_small, M, (vw, vh))   # fixed for every frame — the design/camera mapping doesn't change
    A, t = M[:, :2], M[:, 2]   # used each frame to reposition top relative to bottom (see below)

    # --- Display sizing: 224x224 with --model-resize, else native size downsampled 3x -----------
    def target_size(w: int, h: int) -> tuple[int, int]:
        return (224, 224) if args.model_resize else (w // 8, h // 8)

    # --- Window 1 setup: manual overlay of top (resized for display) on bottom -------------------
    h2_native, w2_native = bottom_full.shape[:2]
    disp1_w, disp1_h = target_size(w2_native, h2_native)
    img1_disp = cv2.resize(top_full, (disp1_w, disp1_h), interpolation=cv2.INTER_AREA)
    img2_disp = cv2.resize(bottom_full, (disp1_w, disp1_h), interpolation=cv2.INTER_AREA)
    h2, w2 = img2_disp.shape[:2]
    h1, w1 = img1_disp.shape[:2]
    start_x, start_y = (w2 - w1) // 2, (h2 - h1) // 2
    off_x, off_y = start_x, start_y

    # top's displacement relative to bottom is set in Window 1's (disp1) resolution — convert it to
    # bottom_small's resolution so it can be applied to the SIFT-registered transform in Window 2.
    bh, bw = bottom_small.shape[:2]
    scale_x, scale_y = bw / w2, bh / h2

    win1, win2, win3, win4 = 'Manual Overlay', 'SIFT Video Align', 'Diff Magnitude', 'Subtractor'
    for w in (win1, win2, win3, win4):
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    cv2.createTrackbar('opacity', win1, 50, 100, lambda _pos: None)

    frame_idx = 0

    def on_frame(pos):
        nonlocal frame_idx
        frame_idx = pos

    cv2.createTrackbar('frame', win2, 0, max(n_frames - 1, 1), on_frame)
    cv2.createTrackbar('top opacity', win2, 50, 100, lambda _pos: None)
    cv2.createTrackbar('bottom opacity', win2, 50, 100, lambda _pos: None)
    cv2.createTrackbar('video opacity', win2, 50, 100, lambda _pos: None)

    sub_frame_idx = 0

    def on_sub_frame(pos):
        nonlocal sub_frame_idx
        sub_frame_idx = pos

    cv2.createTrackbar('frame', win4, 0, max(n_frames - 1, 1), on_sub_frame)

    print("Window 1: WASD/arrows nudge top (also repositions it in Window 2), 'opacity' trackbar blends top/bottom locally.")
    print("Window 2: 'frame' scrubs the video; 'top'/'bottom'/'video opacity' are normalized (sum-to-1) into blend weights. ESC (any window) quits.")
    print("Window 4 (Subtractor): 'frame' picks the frame subtracted from Window 2's in Window 3.")

    out_size = target_size(vw, vh)

    blur_sigma = {'top': 0.0, 'bottom': 0.0, 'video': 0.0}
    blur_target = 'top'   # selected by keys 1/2/3; +/- adjust blur_sigma[blur_target]

    while True:
        opacity = cv2.getTrackbarPos('opacity', win1) / 100.0

        # --- Window 1: manual overlay ---
        M1 = np.array([[1, 0, off_x], [0, 1, off_y]], dtype=np.float32)
        warped1 = cv2.warpAffine(img1_disp, M1, (w2, h2))
        warped1_b = blur_img(warped1, blur_sigma['top'])
        img2_disp_b = blur_img(img2_disp, blur_sigma['bottom'])
        overlay1 = cv2.addWeighted(warped1_b, opacity, img2_disp_b, 1.0 - opacity, 0.0)
        display1 = label(overlay1, [
            f'{Path(args.top).name} over {Path(args.bottom).name}',
            f'displacement: ({off_x - start_x:+d}, {off_y - start_y:+d}) px',
            f'opacity: {opacity:.2f}',
            f'blur (1/2/3 select, +/- adjust)  top={blur_sigma["top"]:.2f} bottom={blur_sigma["bottom"]:.2f} video={blur_sigma["video"]:.2f}  [selected: {blur_target}]',
        ])
        cv2.imshow(win1, display1)

        # --- Window 2: top + bottom, SIFT-registered onto the scrubbed video frame ---
        cv2.setTrackbarPos('frame', win2, frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        top_opacity = cv2.getTrackbarPos('top opacity', win2) / 100.0
        bottom_opacity = cv2.getTrackbarPos('bottom opacity', win2) / 100.0
        video_opacity = cv2.getTrackbarPos('video opacity', win2) / 100.0

        # top's manual offset (relative to bottom, from Window 1) -> bottom_small's coordinate space,
        # then folded into the SIFT-fit transform so top rides along at the same relative position.
        dx, dy = (off_x - start_x) * scale_x, (off_y - start_y) * scale_y
        t_top = t + A @ np.array([dx, dy], dtype=np.float32)
        M_top = np.hstack([A, t_top.reshape(2, 1)]).astype(np.float32)
        warped_top = cv2.warpAffine(top_small, M_top, (vw, vh))

        # normalize the three opacity sliders to weights that sum to 1
        opacity_sum = top_opacity + bottom_opacity + video_opacity
        if opacity_sum > 0:
            w_top, w_bottom, w_video = (top_opacity / opacity_sum, bottom_opacity / opacity_sum, video_opacity / opacity_sum)
        else:
            w_top = w_bottom = w_video = 0.0

        # per-layer blur control (keys 1/2/3 select the layer, +/- adjust its sigma) — resize down to
        # display resolution *before* blurring (like Window 1/4) so a given sigma reads the same way in
        # every window, instead of getting washed out by the ~8x display downsample that follows it
        warped_top_disp = blur_img(cv2.resize(warped_top, out_size, interpolation=cv2.INTER_AREA), blur_sigma['top'])
        warped_bottom_disp = blur_img(cv2.resize(warped_bottom, out_size, interpolation=cv2.INTER_AREA), blur_sigma['bottom'])
        frame_disp = blur_img(cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA), blur_sigma['video'])

        overlay2_disp = (w_top * warped_top_disp.astype(np.float32)
                         + w_bottom * warped_bottom_disp.astype(np.float32)
                         + w_video * frame_disp.astype(np.float32))
        overlay2_disp = np.clip(overlay2_disp, 0, 255).astype(np.uint8)

        display2 = label(overlay2_disp, [
            f'frame {frame_idx}/{n_frames - 1}',
            f'bottom: {Path(args.bottom).name}  top: {Path(args.top).name}  ({out_size[0]}x{out_size[1]})',
            f'opacity  top={top_opacity:.2f} bottom={bottom_opacity:.2f} video={video_opacity:.2f}',
            f'weight   top={w_top:.2f} bottom={w_bottom:.2f} video={w_video:.2f}',
            f'blur (1/2/3 select, +/- adjust)  top={blur_sigma["top"]:.2f} bottom={blur_sigma["bottom"]:.2f} video={blur_sigma["video"]:.2f}  [selected: {blur_target}]',
        ])
        cv2.imshow(win2, display2)

        # --- Window 4: Subtractor — an independently-scrubbed video frame ---
        cv2.setTrackbarPos('frame', win4, sub_frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sub_frame_idx)
        ok_sub, frame_sub = cap.read()
        if not ok_sub:
            frame_sub = np.zeros((vh, vw, 3), dtype=np.uint8)
        frame_sub_disp = blur_img(cv2.resize(frame_sub, out_size), blur_sigma['video'])
        display4 = label(frame_sub_disp, [
            f'frame {sub_frame_idx}/{n_frames - 1}',
            f'blur (3 select, +/- adjust)  video={blur_sigma["video"]:.2f}',
        ])
        cv2.imshow(win4, display4)

        # --- Window 3: diff magnitude between Window 2's overlay (top+bottom on frame) and Window 4's (Subtractor) ---
        diff_mag = np.abs(overlay2_disp.astype(np.float32) - frame_sub_disp.astype(np.float32)).mean(axis=-1)
        diff_mag[diff_mag > MAX_CLIP] = MAX_CLIP
        heat = cv2.applyColorMap(np.clip(diff_mag, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        display3 = label(heat, [f'SIFT align overlay - frame {sub_frame_idx} (Subtractor)', f"Max: {np.amax(diff_mag)} | Median: {np.median(diff_mag)}"])
        cv2.imshow(win3, display3)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:                                        # ESC
            break
        elif key in (ord('d'), 83):                          # right / d
            off_x += args.step
        elif key in (ord('a'), 81):                           # left / a
            off_x -= args.step
        elif key in (ord('s'), 84):                           # down / s
            off_y += args.step
        elif key in (ord('w'), 82):                           # up / w
            off_y -= args.step
        elif key == ord('1'):
            blur_target = 'top'
        elif key == ord('2'):
            blur_target = 'bottom'
        elif key == ord('3'):
            blur_target = 'video'
        elif key in (ord('+'), ord('=')):
            blur_sigma[blur_target] += 0.05
        elif key == ord('-'):
            blur_sigma[blur_target] = max(0.0, blur_sigma[blur_target] - 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
