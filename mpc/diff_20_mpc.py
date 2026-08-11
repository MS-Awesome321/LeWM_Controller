"""
Diff-predictor MPC controller for the nanochemistry transfer stage, using the diff_pred_20x ViT
trained in diff_pred_20x.ipynb (same architecture/checkpoint schema as diff_mpc.py's diff_pred.ipynb
model — a single blank ViT run on the blurred difference image blur(goal) - blur(current), regressing
[Δx, Δy, Δcx, Δcy, Δarea] from its CLS token — but trained on the 20x synthetic top/bottom-composite +
real-pair mix instead of only real episodes).

Unlike diff_mpc.py, the goal frame isn't a pre-rendered image file: at startup this script opens an
interactive 'Goal Composer' window where you position --top over --bottom (WASD/arrows) and set its
opacity (trackbar) — top.jpg/bottom.jpg share the same resolution and pixel grid, so this is a plain
translate + blend, no video or SIFT registration involved. ENTER/'c' bakes the current offset+opacity
into a fixed 224x224 RGB frame, which becomes the fixed target for the MPC loop below.

Control runs in the same two phases as diff_mpc.py (there's no reliable *magnitude* Δz signal in this
model):
  1. 'xy' — move by the predicted Δ(x, y) (mm) each step until within --threshold.
  2. 'z'  — once xy has converged, nudge z by a fixed --z_step each step: if the predicted target ring
            area is greater than the current ring area (Δarea > 0) move z down, otherwise move z up,
            until |Δarea| is within --area_threshold (px²).

Each MPC step: observe → predict → move (xy or z, depending on phase) → re-observe.

Usage:
    python mpc/diff_20_mpc.py                                   # prompts for a 20X_DropDown/<n> folder
    python mpc/diff_20_mpc.py --folder 20X_DropDown/2
    python mpc/diff_20_mpc.py --folder 20X_DropDown/2 --dry_run   # predict only, no robot movement

All distances are in mm (the unit used during training), except Δcx/Δcy/Δarea which are px.
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

DATA_DIR    = REPO_ROOT / '20X_DropDown'
IMG_SIZE    = 224   # ViT input resolution — matches diff_pred_20x.ipynb
PATCH_SIZE  = 16
EMB_DIM     = 192
BLUR_SIGMA  = 2.0    # must match diff_pred_20x.ipynb training
TOP_OPACITY = 0.2    # diff_pred_20x.ipynb's default synthetic-composite opacity — initial trackbar value only
DRY_RUN_FRAME_SIZE = (960, 540)   # placeholder frame size when --dry_run has no camera attached

DEBOUNCE = 20   # loop iterations to skip after issuing a move


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
    diff = gaussian_blur(goal, blur_kernel) - gaussian_blur(cur, blur_kernel)
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


def compose_goal(top_path: Path, bottom_path: Path, step: int) -> np.ndarray:
    """Interactive 'Goal Composer' window: position --top over --bottom (WASD/arrows) and set its
    opacity (trackbar) — ENTER/'c' bakes the current offset+opacity into a fixed IMG_SIZE x IMG_SIZE
    RGB goal frame; ESC cancels the whole program.
    """
    top_full    = cv2.imread(str(top_path))
    bottom_full = cv2.imread(str(bottom_path))
    if top_full is None:
        raise FileNotFoundError(f'Could not read image: {top_path}')
    if bottom_full is None:
        raise FileNotFoundError(f'Could not read image: {bottom_path}')

    h, w = bottom_full.shape[:2]
    preview_w = min(w, 1280)
    preview_h = int(round(h * preview_w / w))

    off_x, off_y = 0.0, 0.0   # (dx, dy) in top.jpg/bottom.jpg's shared native pixel space

    win = 'Goal Composer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, preview_w, preview_h)
    cv2.createTrackbar('opacity', win, int(round(TOP_OPACITY * 100)), 100, lambda _pos: None)

    print(f'Positioning {top_path.name} over {bottom_path.name}.')
    print("WASD/arrows nudge top, 'opacity' trackbar blends it, ENTER/'c' confirms as the MPC goal, ESC cancels.")

    while True:
        opacity = cv2.getTrackbarPos('opacity', win) / 100.0

        M_top = np.array([[1.0, 0.0, off_x], [0.0, 1.0, off_y]], dtype=np.float32)
        warped_top = cv2.warpAffine(top_full, M_top, (w, h))
        composite = cv2.addWeighted(warped_top, opacity, bottom_full, 1.0 - opacity, 0.0)

        display = cv2.resize(composite, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        display = label(display, [
            f'{top_path.name} over {bottom_path.name}',
            f'offset: ({off_x:+.0f}, {off_y:+.0f}) px',
            f'opacity: {opacity:.2f}',
            "WASD/arrows nudge, ENTER/'c' confirm, ESC cancel",
        ])
        cv2.imshow(win, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:                                        # ESC
            cv2.destroyWindow(win)
            raise SystemExit('Cancelled goal composition.')
        elif key in (13, ord('c')):                          # ENTER / c
            break
        elif key in (ord('d'), 83):                          # right / d
            off_x += step
        elif key in (ord('a'), 81):                           # left / a
            off_x -= step
        elif key in (ord('s'), 84):                           # down / s
            off_y += step
        elif key in (ord('w'), 82):                           # up / w
            off_y -= step

    cv2.destroyWindow(win)

    goal_bgr = cv2.resize(composite, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    print(f'Goal composed: offset=({off_x:+.0f}, {off_y:+.0f}) px, opacity={opacity:.2f} '
          f'-> {IMG_SIZE}x{IMG_SIZE} goal frame.')
    return goal_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Live overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame_bgr: np.ndarray, step: int) -> np.ndarray:
    """Live camera feed (at its native, pre-resize resolution) with just the step count top-right."""
    out = frame_bgr.copy()
    w   = out.shape[1]

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thick = 3

    text = f'Step: {step}'
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(out, text, (w - tw - 10, th + 10), font, scale, (0, 220, 255), thick, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='diff_pred_20x MPC controller with an interactive top/bottom goal composer.')
    p.add_argument('--folder', default=None, help='20X_DropDown/<n> folder providing top.jpg/bottom.jpg (prompts if omitted and --top/--bottom are not both given)')
    p.add_argument('--top',    default=None, help='Top image path (default: <folder>/top.jpg)')
    p.add_argument('--bottom', default=None, help='Bottom image path (default: <folder>/bottom.jpg)')
    p.add_argument('--step',   type=int, default=10, help='Pixels nudged per key press when positioning --top over --bottom')
    p.add_argument('--ckpt',   default=str(REPO_ROOT / 'checkpoints' / 'diff_pred_20x_epoch_0099.pt'),
                                              help='Path to a diff_pred_20x checkpoint (.pt). Defaults to the latest '
                                                   'diff_pred_20x_epoch_*.pt in checkpoints/, falling back to '
                                                   'diff_pred_epoch_0040.pt if none exist yet')
    p.add_argument('--dry_run',   action='store_true', help='Predict with dummy frame, no robot/camera')
    p.add_argument('--no_motion', action='store_true', help='Use live camera but skip robot moves')
    p.add_argument('--scale',     type=float, default=1,   help='Fraction of predicted Δ to execute per step')
    p.add_argument('--max_step',  type=float, default=0.1, help='Max |Δ| per axis per step (mm)')
    p.add_argument('--threshold', type=float, default=0.05, help='Goal xy distance threshold (mm)')
    p.add_argument('--area_threshold', type=float, default=100.0,
                                              help='Goal Newton-ring |Δarea| threshold (px²), checked after xy converges')
    p.add_argument('--z_step',   type=float, default=0.01, help='Fixed z nudge per step while aligning ring area (mm)')
    p.add_argument('--debounce', type=int,   default=DEBOUNCE,
                                              help='Loop iterations to skip after a move')
    return p.parse_args()


def main():
    AXES = ('x', 'y', 'z')

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
    ckpt_path = Path(args.ckpt)
    delta_mean, delta_std = load_checkpoint(ckpt_path, vit, head, device)
    blur_kernel = make_gaussian_kernel(BLUR_SIGMA, device)

    # ── goal frame — interactive top/bottom positioning + opacity ───────────────
    goal_rgb = compose_goal(Path(args.top), Path(args.bottom), args.step)

    # ── hardware ──────────────────────────────────────────────────────────────
    cam   = None
    robot = None
    position = []

    try:
        if not args.dry_run:
            from hardware.camera_controller import CameraController
            cam = CameraController(index=0, fps=15)
            cam.start()

        if not args.dry_run and not args.no_motion:
            from hardware.transfer_control_controller import TransferControl
            robot = TransferControl(only_xyz=True)
            robot.connect()
            p = robot.positions()
            print('Robot connected. Positions:', p)
            position = [float(pos) for _, pos in robot.positions().items()]

        if args.no_motion:
            print('No-motion mode — camera live, moves skipped.')

        # ── cv2 loop ──────────────────────────────────────────────────────────
        cv2.namedWindow('Diff-20x-MPC', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Diff-20x-MPC', DRY_RUN_FRAME_SIZE[0], DRY_RUN_FRAME_SIZE[1])

        step          = 0
        debounce_i    = 0
        dist          = float('inf')
        phase         = 'xy'   # 'xy' -> 'z'

        while True:
            if cam is not None:
                frame_bgr = cam.snap()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = np.zeros((DRY_RUN_FRAME_SIZE[1], DRY_RUN_FRAME_SIZE[0], 3), dtype=np.uint8)
                frame_bgr = frame_rgb.copy()

            display = draw_overlay(frame_bgr, step)
            cv2.imshow('Diff-20x-MPC', display)
            if cv2.waitKey(1) == 27:   # ESC
                print('ESC — stopping.')
                break

            # ── debounce ──────────────────────────────────────────────────────
            if debounce_i > 0:
                debounce_i -= 1
                continue

            if robot is not None:
                kst_axes = [robot._get_axis(ax) for ax in AXES]
                if any(a.dev.Status.IsMoving for a in kst_axes):
                    continue

            # ── observe & predict (only when ready) ───────────────────────────
            pred_full = predict_displacement(frame_rgb, goal_rgb, vit, head, blur_kernel,
                                             delta_mean, delta_std, device)
            pred_xy   = pred_full[:2]
            pred_area = float(pred_full[4])
            dist      = float(np.linalg.norm(pred_xy))

            if phase == 'xy':
                if dist < args.threshold:
                    print(f'xy converged ({dist:.4f} mm away) — switching to z alignment.')
                    phase = 'z'
                    continue

                # ── scale & clamp move (x, y only — no Δz magnitude signal) ───
                move_xy = np.clip(pred_xy * args.scale, -args.max_step, args.max_step)

                if robot is not None:
                    for i, ax in enumerate(('x', 'y')):
                        position[i] += float(move_xy[i])
                        robot.move_axis_to(ax, position[i])
                    print(f'  move x to {position[0]:+.4f} mm, y to {position[1]:+.4f} mm')
                    debounce_i = args.debounce
                else:
                    print(f'  move x by {move_xy[0]:+.4f} mm, y by {move_xy[1]:+.4f} mm  (no-op)')

            else:   # phase == 'z'
                if abs(pred_area) < args.area_threshold:
                    print(f'Goal reached. xy={dist:.4f} mm away, |Δarea|={abs(pred_area):.2f} px² away.')
                    break

                # target ring area greater than current -> move down; otherwise up.
                move_z = -args.z_step if pred_area > 0 else args.z_step

                if robot is not None:
                    position[2] += move_z
                    print(f'  move z to {position[2]:+.4f} mm  (Δarea={pred_area:+.2f} px²)')
                    robot.move_axis_to('z', position[2])
                    debounce_i = args.debounce
                else:
                    print(f'  move z by {move_z:+.4f} mm  (Δarea={pred_area:+.2f} px²)  (no-op)')

            step += 1

    except KeyboardInterrupt:
        print('\nInterrupted.')

    except Exception as e:
        print(f'Error: {e}')
        raise

    finally:
        if robot is not None:
            robot.disconnect()
            print('Done.')
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
