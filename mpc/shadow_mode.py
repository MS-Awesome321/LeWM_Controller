"""
Shadow mode: manually jog the stage (same controls as manual_control.py) while the diff_pred_20x ViT
continuously watches the live camera feed and predicts the displacement to a fixed goal frame, composed
the same way as in diff_20_mpc.py (position --top over --bottom, WASD/arrows + opacity trackbar, ENTER/'c'
to confirm). The predicted displacement is shown on screen next to the "real" displacement — the actual
mm moved since shadow mode started, read straight from the motor positions — so you can manually drive the
stage and eyeball how well the model's predictions track real motion. No robot moves are ever issued by
this script beyond your own key presses.

The 'Goal Composer' window stays open for the whole program (not just at startup), so you can change the
goal at any time. OpenCV doesn't report which window is focused, so WASD/arrows can't be routed by "which
window you clicked into" — instead press 'g' to toggle edit mode: while editing, WASD/arrows nudge the
composer and stage jogging is suspended; ENTER/'c' bakes the new goal and returns to jogging.

Controls (identical to manual_control.py, except 'g'):
    w/a/s/d — jog y+/x+/y-/x- (or, in goal-edit mode, nudge the composer overlay)
    q/e     — jog z+/z-
    g       — toggle goal-edit mode
    ENTER/c — (goal-edit mode only) bake the composer's current offset/opacity into the goal
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
TOP_OPACITY = 0.2    # diff_pred_20x.ipynb's default synthetic-composite opacity — initial trackbar value only

DEBOUNCE = 5   # loop iterations to skip after a jog key press (same scheme as manual_control.py)


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


def pick_folder() -> Path:
    folders = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and (p / 'top.jpg').exists() and (p / 'bottom.jpg').exists())
    if not folders:
        raise FileNotFoundError(f'No folders with top.jpg/bottom.jpg found under {DATA_DIR}')
    print('Available folders (top.jpg/bottom.jpg pairs):')
    for idx, path in enumerate(folders):
        print(f'  [{idx}] {path.name}')
    choice = input(f'Pick a folder [0-{len(folders) - 1}]: ').strip()
    return folders[int(choice)]


def init_composer(top_path: Path, bottom_path: Path) -> dict:
    """Load top/bottom and open the 'Goal Composer' window (with its opacity trackbar). The window
    and returned state stay alive for the whole program, so the goal can be re-baked at any time —
    see render_composer()/nudge_composer()/bake_goal()."""
    top_full    = cv2.imread(str(top_path))
    bottom_full = cv2.imread(str(bottom_path))
    if top_full is None:
        raise FileNotFoundError(f'Could not read image: {top_path}')
    if bottom_full is None:
        raise FileNotFoundError(f'Could not read image: {bottom_path}')

    h, w = bottom_full.shape[:2]
    preview_w = min(w, 1280)
    preview_h = int(round(h * preview_w / w))

    win = 'Goal Composer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, preview_w, preview_h)
    cv2.createTrackbar('opacity', win, int(round(TOP_OPACITY * 100)), 100, lambda _pos: None)

    return {
        'win': win, 'top_path': top_path, 'bottom_path': bottom_path,
        'top_full': top_full, 'bottom_full': bottom_full,
        'h': h, 'w': w, 'preview_w': preview_w, 'preview_h': preview_h,
        'off_x': 0.0, 'off_y': 0.0, 'composite': bottom_full.copy(),
    }


def render_composer(c: dict, editing: bool) -> np.ndarray:
    """Recompute the composite from c's current offset + the opacity trackbar and redraw the
    composer window. Returns the full-resolution BGR composite (bake_goal() turns it into a goal)."""
    opacity = cv2.getTrackbarPos('opacity', c['win']) / 100.0
    M = np.array([[1.0, 0.0, c['off_x']], [0.0, 1.0, c['off_y']]], dtype=np.float32)
    warped_top = cv2.warpAffine(c['top_full'], M, (c['w'], c['h']))
    composite = cv2.addWeighted(warped_top, opacity, c['bottom_full'], 1.0 - opacity, 0.0)
    c['composite'] = composite

    display = cv2.resize(composite, (c['preview_w'], c['preview_h']), interpolation=cv2.INTER_AREA)
    display = label(display, [
        f"{c['top_path'].name} over {c['bottom_path'].name}",
        f'offset: ({c["off_x"]:+.0f}, {c["off_y"]:+.0f}) px   opacity: {opacity:.2f}',
        "EDITING - WASD/arrows nudge, ENTER/'c' bake, g to lock" if editing else "locked - press g to edit",
    ])
    cv2.imshow(c['win'], display)
    return composite


def nudge_composer(c: dict, key: int, step: int) -> None:
    if key in (ord('d'), 83):                          # right / d
        c['off_x'] += step
    elif key in (ord('a'), 81):                          # left / a
        c['off_x'] -= step
    elif key in (ord('s'), 84):                          # down / s
        c['off_y'] += step
    elif key in (ord('w'), 82):                          # up / w
        c['off_y'] -= step


def bake_goal(composite: np.ndarray) -> np.ndarray:
    goal_bgr = cv2.resize(composite, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Manually jog the stage while shadowing diff_pred_20x predictions against real motor motion.')
    p.add_argument('--folder', default=None, help='20X_DropDown/<n> folder providing top.jpg/bottom.jpg (prompts if omitted and --top/--bottom are not both given)')
    p.add_argument('--top',    default=None, help='Top image path (default: <folder>/top.jpg)')
    p.add_argument('--bottom', default=None, help='Bottom image path (default: <folder>/bottom.jpg)')
    p.add_argument('--step',   type=int, default=10, help='Pixels nudged per key press when positioning --top over --bottom')
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

    # ── goal composer — stays open for the whole program, not just at startup ───
    c = init_composer(Path(args.top), Path(args.bottom))
    print(f'Positioning {c["top_path"].name} over {c["bottom_path"].name}.')
    print("WASD/arrows nudge top, 'opacity' trackbar blends it, ENTER/'c' confirms the initial goal, ESC cancels.")
    while True:
        render_composer(c, editing=True)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                                        # ESC
            cv2.destroyAllWindows()
            raise SystemExit('Cancelled goal composition.')
        elif key in (13, ord('c')):                          # ENTER / c
            break
        else:
            nudge_composer(c, key, args.step)
    goal_rgb = bake_goal(c['composite'])
    print(f'Goal composed: offset=({c["off_x"]:+.0f}, {c["off_y"]:+.0f}) px -> {IMG_SIZE}x{IMG_SIZE} goal frame.')

    cam = None
    arm = None
    try:
        cam = CameraController(index=0, fps=15)
        cam.start()
        cv2.namedWindow('Shadow Mode', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Shadow Mode', 960, 540)

        arm = TransferControl(only_xyz=True)
        ref_pos = arm.positions()
        print('Robot connected. Reference positions:', ref_pos)

        i = 0
        editing_goal = False
        while True:
            frame_bgr = cam.snap()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            pred = predict_displacement(frame_rgb, goal_rgb, vit, head, blur_kernel,
                                        delta_mean, delta_std, device)
            cur_pos = arm.positions()
            real = {axis: cur_pos[axis] - ref_pos[axis] for axis in ('x', 'y', 'z')}

            preview = cv2.resize(frame_bgr, (960, 540), interpolation=cv2.INTER_AREA)
            display = label(preview, [
                f'Predicted (cur->goal): dx={pred[0]:+.4f}mm dy={pred[1]:+.4f}mm  dcx={pred[2]:+.2f}px dcy={pred[3]:+.2f}px darea={pred[4]:+.2f}px2',
                f'Real (since start):    dx={real["x"]:+.4f}mm dy={real["y"]:+.4f}mm dz={real["z"]:+.4f}mm',
            ])
            cv2.imshow('Shadow Mode', display)
            render_composer(c, editing=editing_goal)

            key = cv2.waitKey(1)

            if key == 27:   # ESC — always quits, whether editing the goal or jogging
                print("Stopping All")
                if not editing_goal:
                    arm.stop_xyz()
                break
            elif key == ord('g'):
                editing_goal = not editing_goal
                if editing_goal:
                    arm.stop_xyz()
                print('Editing goal (WASD/arrows nudge, ENTER/c bake)' if editing_goal else 'Goal locked')
            elif editing_goal:
                if key in (13, ord('c')):
                    goal_rgb = bake_goal(c['composite'])
                    editing_goal = False
                    print(f'Goal updated: offset=({c["off_x"]:+.0f}, {c["off_y"]:+.0f}) px.')
                else:
                    nudge_composer(c, key, args.step)
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
