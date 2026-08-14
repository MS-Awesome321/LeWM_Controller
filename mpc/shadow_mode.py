"""
Shadow mode: manually jog the stage (same controls as manual_control.py) while the diff_pred_20x ViT
continuously watches the live camera feed and predicts the displacement to a goal frame composed from
--top over --bottom (WASD/arrows + a draggable opacity slider), with --bottom SIFT-registered
onto the live camera's coordinate frame (same registration pattern as flake_designer.py) so the goal is
always expressed in the camera's own field of view. The predicted displacement is shown on screen next to
the "real" displacement — the actual mm moved since a reference motor position, read straight from the
motor positions — so you can manually drive the stage and eyeball how well the model's predictions track
real motion. The reference position starts out as wherever the stage was on connect, and can be reset to
the current position at any time with 'm'. No robot moves are ever issued by this script beyond your own
key presses.

Everything lives in one window: the live camera feed, the goal (top+bottom, positioned live and SIFT-warped
onto the live view's frame), and a heatmap of |blur(current) - blur(goal)| — the same diff the ViT actually
regresses on — side by side, with predicted vs. real displacement printed underneath. The goal is fully live:
every one of these three panels, and the ViT's input, is recomputed from scratch every loop iteration
directly from the current top/bottom offsets — there's no "confirm"/"bake" step, exactly like the live
camera view itself.

WASD/arrows are shared between jogging the stage and nudging the goal composite, since OpenCV can't tell
which panel you "clicked into" — press 'g' to nudge --top's position over --bottom, or 'h' to nudge
--bottom's position (a manual offset on top of the one-time SIFT registration onto the live view); either
one suspends stage jogging until you press it again to go back to jog mode.

Controls (identical to manual_control.py, except 'g'/'h'/'m'):
    w/a/s/d — jog y+/x+/y-/x- (or, in an edit mode, nudge the top/bottom position)
    q/e     — jog z+/z-
    g       — toggle "edit top" mode (WASD/arrows move --top over --bottom)
    h       — toggle "edit bottom" mode (WASD/arrows move --bottom's SIFT-registered position)
    m       — set the current motor position as the reference point for "Real (since start)"
    0       — save the current camera frame to images/capture_{x}_{y}_{z}.png
    p       — print current motor positions
    ESC     — stop all axes and quit

Usage:
    python mpc/shadow_mode.py                                 # prompts for a 20X_DropDown/<n> folder
    python mpc/shadow_mode.py --folder 20X_DropDown/2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import ViTModel, ViTConfig
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import hardware` resolves regardless of cwd

from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

DATA_DIR    = REPO_ROOT / '20X_DropDown'
IMG_SIZE    = 224   # ViT input resolution — matches diff_pred_20x.ipynb
PATCH_SIZE  = 16
EMB_DIM     = 192
BLUR_SIGMA  = 2.0    # must match diff_pred_20x.ipynb training
TOP_OPACITY = 0.2    # diff_pred_20x.ipynb's default synthetic-composite opacity — initial slider value only
MAX_CLIP    = 60     # diff-mag display clip (0-255 scale) — matches diff_pred_20x.ipynb's demo-cell visualization clip; display only, never applied to the ViT's actual input

DEBOUNCE = 15   # loop iterations to skip after a jog key press (same scheme as manual_control.py)

PANEL_H       = 420   # height of each of the LIVE/GOAL/DIFF panels
SLIDER_H      = 30    # height of the custom opacity-slider strip drawn between the panels and the status strip
SLIDER_MARGIN = 10    # left/right margin (px) of the draggable track within the slider strip


# ─────────────────────────────────────────────────────────────────────────────
# Model definition (must match diff_pred_20x.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────

class DisplacementHead(nn.Module):
    """CLS token -> small MLP -> (5,): [Δx, Δy, Δcx, Δcy, Δarea]."""
    def __init__(self, emb_dim=EMB_DIM, target_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, target_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


def build_model(device: str):
    vit_cfg = ViTConfig(
        num_channels=3, image_size=IMG_SIZE, patch_size=PATCH_SIZE,
        hidden_size=EMB_DIM, num_hidden_layers=6,
        num_attention_heads=3, intermediate_size=768,
    )
    vit  = ViTModel(vit_cfg, add_pooling_layer=False).to(device)
    head = DisplacementHead(emb_dim=EMB_DIM).to(device)
    vit.eval()
    head.eval()
    return vit, head


def load_checkpoint(ckpt_path: Path, vit, head, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vit.load_state_dict(ckpt['vit'])
    head.load_state_dict(ckpt['head'])
    delta_mean = ckpt['delta_mean'].to(device)
    delta_std  = ckpt['delta_std'].to(device)
    print(f'Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get("epoch", "?")})')
    return delta_mean, delta_std


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gaussian_kernel(sigma: float, device: str):
    """(1, 1, k, k) normalized 2D gaussian kernel, k = 2*round(3*sigma)+1."""
    if sigma <= 0:
        return None
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])


def gaussian_blur(x: torch.Tensor, kernel: torch.Tensor | None) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H, W), blurred per-channel (depthwise conv, same padding)."""
    if kernel is None:
        return x
    c = x.shape[1]
    k = kernel.repeat(c, 1, 1, 1).to(dtype=x.dtype, device=x.device)
    pad = k.shape[-1] // 2
    return torch.nn.functional.conv2d(x, k, padding=pad, groups=c)


def to_float_frame(frame_rgb: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (1, 3, IMG_SIZE, IMG_SIZE) float32 [0, 1] on device."""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def predict_displacement(cur_rgb: np.ndarray, goal_rgb: np.ndarray, vit, head,
                         blur_kernel, delta_mean: torch.Tensor, delta_std: torch.Tensor,
                         device: str) -> np.ndarray:
    """Predicted [Δx, Δy, Δcx, Δcy, Δarea] (mm, mm, px, px, px²) to move from cur → goal."""
    cur  = to_float_frame(cur_rgb, device)
    goal = to_float_frame(goal_rgb, device)
    diff = gaussian_blur(cur, blur_kernel) - gaussian_blur(goal, blur_kernel)
    cls_token = vit(diff, interpolate_pos_encoding=False).last_hidden_state[:, 0]
    pred_norm = head(cls_token)
    pred_raw  = pred_norm * delta_std + delta_mean
    return pred_raw.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Goal composer — interactive top/bottom positioning + opacity (top.jpg/bottom.jpg are already the
# same resolution and pixel-aligned, so this is a plain translate + blend)
# ─────────────────────────────────────────────────────────────────────────────

def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (5, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def sift_affine(small_gray: np.ndarray, large_gray: np.ndarray, ratio_thresh: float) -> np.ndarray:
    """SIFT + ratio-test pipeline, restricted to a translation-only fit (no rotation/scale/shear):
    the translation is the median of matched keypoints' displacement vectors, robust to outlier matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(small_gray, None)
    kp2, des2 = sift.detectAndCompute(large_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    if len(good) < 3:
        raise RuntimeError(f'Only {len(good)} good matches — need at least 3 to fit a transform.')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    tx, ty = np.median(dst_pts - src_pts, axis=0)
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return M


def pick_folder() -> Path:
    folders = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and (p / 'top.jpg').exists() and (p / 'bottom.jpg').exists())
    if not folders:
        raise FileNotFoundError(f'No folders with top.jpg/bottom.jpg found under {DATA_DIR}')
    print('Available folders (top.jpg/bottom.jpg pairs):')
    for idx, path in enumerate(folders):
        print(f'  [{idx}] {path.name}')
    choice = input(f'Pick a folder [0-{len(folders) - 1}]: ').strip()
    return folders[int(choice)]


def init_composer(top_path: Path, bottom_path: Path, win: str) -> dict:
    """Load top/bottom. Opacity is driven by a custom mouse-dragged slider (see draw_opacity_slider()/
    opacity_mouse_callback()) rather than cv2.createTrackbar, since native OS trackbars snap to a small
    number of discrete positions regardless of the `count` passed to createTrackbar. The returned state
    stays alive for the whole program and is recomputed every frame — see
    update_composite()/nudge_xy()/warp_goal_to_live()/bake_goal()."""
    top_full    = cv2.imread(str(top_path))
    bottom_full = cv2.imread(str(bottom_path))
    if top_full is None:
        raise FileNotFoundError(f'Could not read image: {top_path}')
    if bottom_full is None:
        raise FileNotFoundError(f'Could not read image: {bottom_path}')

    h, w = bottom_full.shape[:2]

    return {
        'win': win, 'top_path': top_path, 'bottom_path': bottom_path,
        'top_full': top_full, 'bottom_full': bottom_full,
        'h': h, 'w': w,
        'off_x': 0.0, 'off_y': 0.0, 'composite': bottom_full.copy(),
        'bx': 0.0, 'by': 0.0,   # manual offset of --bottom on top of its SIFT-registered position
        'opacity': TOP_OPACITY,   # continuous [0, 1] float, dragged via the custom slider
    }


def update_composite(c: dict) -> np.ndarray:
    """Recompute c['composite'] from its current offset + opacity. Returns the full-resolution BGR
    composite (bake_goal() turns it into a goal)."""
    opacity = c['opacity']
    M = np.array([[1.0, 0.0, c['off_x']], [0.0, 1.0, c['off_y']]], dtype=np.float32)
    warped_top = cv2.warpAffine(c['top_full'], M, (c['w'], c['h']))
    composite = cv2.addWeighted(warped_top, opacity, c['bottom_full'], 1.0 - opacity, 0.0)
    c['composite'] = composite
    c['opacity'] = opacity
    return composite


def nudge_xy(state: dict, key: int, step: int, x_key: str, y_key: str) -> None:
    """Nudge state[x_key]/state[y_key] by WASD/arrows — used for both --top's offset over --bottom
    and --bottom's manual offset on top of its SIFT-registered position."""
    if key in (ord('d'), 83):                          # right / d
        state[x_key] += step
    elif key in (ord('a'), 81):                          # left / a
        state[x_key] -= step
    elif key in (ord('s'), 84):                          # down / s
        state[y_key] += step
    elif key in (ord('w'), 82):                          # up / w
        state[y_key] -= step


def warp_goal_to_live(composite: np.ndarray, sift_M: np.ndarray, bx: float, by: float,
                      live_w: int, live_h: int) -> np.ndarray:
    """Warp the (--bottom-space) composite onto the live camera's coordinate frame using the one-time
    SIFT fit, folding in the manual --bottom offset (bx, by) the same way flake_designer.py folds --top's
    offset into its SIFT fit."""
    A, t = sift_M[:, :2], sift_M[:, 2]
    t_adj = t + A @ np.array([bx, by], dtype=np.float32)
    M_adj = np.hstack([A, t_adj.reshape(2, 1)]).astype(np.float32)
    return cv2.warpAffine(composite, M_adj, (live_w, live_h))


def bake_goal(warped_goal: np.ndarray) -> np.ndarray:
    goal_bgr = cv2.resize(warped_goal, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Diff-mag visualization + unified layout
# ─────────────────────────────────────────────────────────────────────────────

def diff_mag_image(cur_rgb: np.ndarray, goal_rgb: np.ndarray) -> np.ndarray:
    """|blur(cur) - blur(goal)| — the same blurred diff the ViT regresses on — as an IMG_SIZE x
    IMG_SIZE BGR heatmap (cv2 GaussianBlur stand-in for the torch depthwise-conv blur used at
    inference time, since this is for display only). Clipped to MAX_CLIP (0-255 scale) before
    colormapping, matching diff_pred_20x.ipynb's demo-cell visualization — the ViT's actual input
    (predict_displacement()'s diff tensor) is never clipped, matching the notebook's real
    extract_embedding() path."""
    cur  = cv2.resize(cur_rgb,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    goal = cv2.resize(goal_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    k = 2 * max(1, int(round(3 * BLUR_SIGMA))) + 1
    cur_b  = cv2.GaussianBlur(cur,  (k, k), BLUR_SIGMA)
    goal_b = cv2.GaussianBlur(goal, (k, k), BLUR_SIGMA)
    mag_255 = np.mean(np.abs(cur_b - goal_b), axis=2) * 255.0
    mag_clipped = np.clip(mag_255, 0, MAX_CLIP)
    mag_u8 = (mag_clipped / MAX_CLIP * 255.0).astype(np.uint8)
    return cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)


def draw_opacity_slider(width: int, opacity: float) -> np.ndarray:
    """Custom continuous opacity slider (replaces cv2.createTrackbar — see init_composer()),
    drawn as a SLIDER_H x width strip with a filled track and a handle at `opacity`'s position.
    opacity_mouse_callback() maps mouse x back to a [0, 1] float using the same SLIDER_MARGIN."""
    strip = np.full((SLIDER_H, width, 3), 40, dtype=np.uint8)
    usable = max(1, width - 2 * SLIDER_MARGIN)
    track_y = SLIDER_H // 2
    cv2.line(strip, (SLIDER_MARGIN, track_y), (width - SLIDER_MARGIN, track_y), (100, 100, 100), 2, cv2.LINE_AA)
    handle_x = SLIDER_MARGIN + int(round(opacity * usable))
    cv2.line(strip, (SLIDER_MARGIN, track_y), (handle_x, track_y), (0, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(strip, (handle_x, track_y), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(strip, f'opacity: {opacity:.3f}  (drag)', (SLIDER_MARGIN, SLIDER_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return strip


def opacity_mouse_callback(event, x, y, flags, c: dict) -> None:
    """Registered once via cv2.setMouseCallback(WIN, opacity_mouse_callback, c). Dragging (left button
    down or held) within the slider strip sets c['opacity'] continuously — see draw_opacity_slider()
    for the matching layout/geometry."""
    if not (0 <= y - c['slider_y0'] < SLIDER_H):
        return
    if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
        usable = max(1, c['canvas_w'] - 2 * SLIDER_MARGIN)
        frac = (x - SLIDER_MARGIN) / usable
        c['opacity'] = float(np.clip(frac, 0.0, 1.0))


def compose_view(live_bgr: np.ndarray, goal_bgr: np.ndarray, diff_bgr: np.ndarray,
                 mode: str, pred: np.ndarray, real: dict, c: dict) -> np.ndarray:
    """Combine the live feed, SIFT-warped goal, and diff-mag heatmap into one canvas, with a custom
    opacity-slider strip and a status strip (mode + predicted/real displacement) underneath."""

    def fit(img):
        ih, iw = img.shape[:2]
        w = max(1, int(round(iw * PANEL_H / ih)))
        return cv2.resize(img, (w, PANEL_H), interpolation=cv2.INTER_AREA)

    live_p = label(fit(live_bgr), ['LIVE'])
    goal_p = label(fit(goal_bgr), [
        'GOAL (SIFT-aligned to live)',
        f'top=({c["off_x"]:+.0f},{c["off_y"]:+.0f}) op={c.get("opacity", 0.0):.2f}',
        f'bottom=({c["bx"]:+.0f},{c["by"]:+.0f})',
    ])
    diff_p = label(fit(diff_bgr), ['DIFF MAG |cur-goal|'])

    top = np.hstack([live_p, goal_p, diff_p])

    # opacity_mouse_callback() reads these back to know where the slider strip is in canvas coords
    c['canvas_w']  = top.shape[1]
    c['slider_y0'] = PANEL_H
    slider = draw_opacity_slider(top.shape[1], c.get('opacity', TOP_OPACITY))

    status = np.zeros((70, top.shape[1], 3), dtype=np.uint8)
    mode_line = {
        'edit_top':    "EDIT TOP - WASD/arrows move top over bottom, g to lock",
        'edit_bottom': "EDIT BOTTOM - WASD/arrows move bottom's SIFT-registered position, h to lock",
        'jog':         "JOG MODE - w/a/s/d/q/e move stage, g to edit top, h to edit bottom, m to set reference",
    }[mode]
    status = label(status, [
        mode_line,
        f'Predicted (cur->goal): dx={pred[0]:+.4f}mm dy={pred[1]:+.4f}mm  dcx={pred[2]:+.2f}px dcy={pred[3]:+.2f}px darea={pred[4]:+.2f}px2',
        f'Real (since ref, m to reset): dx={real["x"]:+.4f}mm dy={real["y"]:+.4f}mm dz={real["z"]:+.4f}mm',
    ])

    return np.vstack([top, slider, status])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Manually jog the stage while shadowing diff_pred_20x predictions against real motor motion.')
    p.add_argument('--folder', default=None, help='20X_DropDown/<n> folder providing top.jpg/bottom.jpg (prompts if omitted and --top/--bottom are not both given)')
    p.add_argument('--top',    default=None, help='Top image path (default: <folder>/top.jpg)')
    p.add_argument('--bottom', default=None, help='Bottom image path (default: <folder>/bottom.jpg)')
    p.add_argument('--step',   type=int, default=10, help='Pixels nudged per key press when positioning --top over --bottom, or --bottom over the live view')
    p.add_argument('--ratio',  type=float, default=0.75, help="Lowe's ratio test threshold for the one-time SIFT registration of --bottom onto the live view")
    p.add_argument('--ckpt',   default=str(REPO_ROOT / 'checkpoints' / 'diff_pred_20x_epoch_0099.pt'),
                                              help='Path to a diff_pred_20x checkpoint (.pt)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.top is None or args.bottom is None:
        folder = Path(args.folder) if args.folder else pick_folder()
        if args.top is None:
            args.top = str(folder / 'top.jpg')
        if args.bottom is None:
            args.bottom = str(folder / 'bottom.jpg')

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── model ─────────────────────────────────────────────────────────────────
    vit, head = build_model(device)
    delta_mean, delta_std = load_checkpoint(Path(args.ckpt), vit, head, device)
    blur_kernel = make_gaussian_kernel(BLUR_SIGMA, device)

    # ── unified window — live feed, goal composite, and diff-mag heatmap all live here ──────────
    WIN = 'Shadow Mode'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 500)

    c = init_composer(Path(args.top), Path(args.bottom), WIN)
    c['canvas_w'], c['slider_y0'] = 1, 0   # placeholder — compose_view() fills these in before the slider can be dragged
    cv2.setMouseCallback(WIN, opacity_mouse_callback, c)
    print(f'Positioning {c["top_path"].name} over {c["bottom_path"].name}. Press g to move top, h to move bottom.')

    cam = None
    arm = None
    try:
        cam = CameraController(index=0, fps=15)
        cam.start()

        # One-time SIFT registration of --bottom onto the live camera's coordinate frame (same
        # pattern as flake_designer.py) — features detected on a blurred copy, fit reused every frame.
        ref_frame_bgr = cam.snap()
        bottom_blurred = cv2.GaussianBlur(c['bottom_full'], (15, 15), 0)
        sift_M = sift_affine(cv2.cvtColor(bottom_blurred, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY), args.ratio)
        print('SIFT-registered bottom onto the live view. Press h to adjust its position manually.')

        arm = TransferControl(only_xyz=True)
        ref_pos = arm.positions()
        print('Robot connected. Reference positions:', ref_pos)

        i = 0
        mode = 'jog'   # 'jog' | 'edit_top' | 'edit_bottom'
        while True:
            frame_bgr = cam.snap()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            live_h, live_w = frame_bgr.shape[:2]

            # goal is recomputed fresh every iteration — no bake/confirm step, exactly like the live view
            composite   = update_composite(c)
            warped_goal = warp_goal_to_live(composite, sift_M, c['bx'], c['by'], live_w, live_h)
            goal_rgb    = bake_goal(warped_goal)

            pred = predict_displacement(frame_rgb, goal_rgb, vit, head, blur_kernel,
                                        delta_mean, delta_std, device)
            cur_pos = arm.positions()
            real = {axis: cur_pos[axis] - ref_pos[axis] for axis in ('x', 'y', 'z')}

            diff_img = diff_mag_image(frame_rgb, goal_rgb)
            canvas   = compose_view(frame_bgr, warped_goal, diff_img, mode, pred, real, c)
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(1)

            if key == 27:   # ESC — always quits, whether editing the goal or jogging
                print("Stopping All")
                if mode == 'jog':
                    arm.stop_xyz()
                break
            elif key == ord('g'):
                mode = 'jog' if mode == 'edit_top' else 'edit_top'
                if mode != 'jog':
                    arm.stop_xyz()
                print('Editing top (WASD/arrows move it over bottom)' if mode == 'edit_top' else 'Back to jog mode')
            elif key == ord('h'):
                mode = 'jog' if mode == 'edit_bottom' else 'edit_bottom'
                if mode != 'jog':
                    arm.stop_xyz()
                print("Editing bottom (WASD/arrows adjust its SIFT-registered position)" if mode == 'edit_bottom' else 'Back to jog mode')
            elif key == ord('m'):
                ref_pos = arm.positions()
                print('Reference position set:', ref_pos)
            elif mode == 'edit_top':
                nudge_xy(c, key, args.step, 'off_x', 'off_y')
            elif mode == 'edit_bottom':
                nudge_xy(c, key, args.step, 'bx', 'by')
            elif i >= 0:
                if key == 123 or key == 97:   # a
                    print('Left')
                    i = -DEBOUNCE
                    arm.jog_axis('x', '+')
                elif key == 124 or key == 100:   # d
                    print('Right')
                    i = -DEBOUNCE
                    arm.jog_axis('x', '-')
                elif key == 125 or key == 115:   # s
                    print('Down')
                    i = -DEBOUNCE
                    arm.jog_axis('y', '-')
                elif key == 126 or key == 119:   # w
                    print('Up')
                    i = -DEBOUNCE
                    arm.jog_axis('y', '+')
                elif key == 113:   # q
                    print('Raise')
                    i = -DEBOUNCE
                    arm.jog_axis('z', '+')
                elif key == 101:   # e
                    print('Lower')
                    i = -DEBOUNCE
                    arm.jog_axis('z', '-')
                elif key == 48:   # 0
                    pos = arm.positions()
                    fname = f"images/capture_{pos['x']}_{pos['y']}_{pos['z']}.png"
                    cv2.imwrite(fname, frame_bgr)
                    print(f'Saved {fname}')
                elif key == 112:   # p
                    print(arm.positions())
                else:
                    arm.stop_xyz()
            elif i >= -2*DEBOUNCE//3:
                arm.stop_xyz()

            i += 1

    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.stop()
        if arm is not None:
            arm.disconnect()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
