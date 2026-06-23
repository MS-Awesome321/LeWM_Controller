"""
Manual keyboard control for the nanochemistry transfer stage with live camera view.

Keys:
    Arrow Up    — move Y forward
    Arrow Down  — move Y backward
    Arrow Left  — move X left
    Arrow Right — move X right
    Q           — move Z up
    E           — move Z down
    =           — increase ACTION_SCALE by 0.1
    -           — decrease ACTION_SCALE by 0.1 (min 0.1)
    ESC / Ctrl-C — home and exit
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

# ── constants ─────────────────────────────────────────────────────────────────
STEP_MM       = 1.0    # base step size in mm
ACTION_SCALE  = 0.1    # multiplied against STEP_MM; change with = / -
SCALE_STEP    = 0.1    # how much = / - adjusts ACTION_SCALE
SCALE_MIN     = 0.1
SCALE_MAX     = 10.0
WIN_TITLE     = 'Manual Transfer Control'
WIN_W, WIN_H  = 800, 600

# cv2 special key codes (platform-dependent; these work on Windows and Linux)
KEY_UP    = 2490368   # Windows
KEY_DOWN  = 2621440
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904
KEY_UP_L  = 82        # Linux / macOS (lowercase)
KEY_DOWN_L  = 84
KEY_LEFT_L  = 81
KEY_RIGHT_L = 83


def parse_args():
    p = argparse.ArgumentParser(description='Manual transfer stage controller.')
    p.add_argument('--fps',      type=int,   default=15,  help='Camera FPS')
    p.add_argument('--index',    type=int,   default=0,   help='Camera device index')
    p.add_argument('--step',     type=float, default=STEP_MM, help='Base step size (mm)')
    p.add_argument('--no_robot', action='store_true', help='Camera only, no robot connection')
    return p.parse_args()


# ── camera thread ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Grabs frames from the camera and exposes the latest one thread-safely."""

    def __init__(self, camera: CameraController):
        super().__init__(daemon=True)
        self._camera   = camera
        self._lock     = threading.Lock()
        self._frame    = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                bgr = self._camera.snap()
                with self._lock:
                    self._frame = bgr
            except Exception:
                pass

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def stop(self):
        self._stop_evt.set()


# ── overlay ───────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, scale: float, cumulative: dict, step_mm: float) -> np.ndarray:
    canvas = cv2.resize(frame, (WIN_W, WIN_H))
    effective = step_mm * scale
    lines = [
        f'ACTION_SCALE: {scale:.1f}  (step = {effective:.3f} mm)',
        f'Cumulative:  x={cumulative["x"]:+.3f}  y={cumulative["y"]:+.3f}  z={cumulative["z"]:+.3f} mm',
        'Arrows=XY  Q/E=Z  =/- scale  ESC=home+quit',
    ]
    for i, text in enumerate(lines):
        y = WIN_H - 20 - (len(lines) - 1 - i) * 22
        cv2.putText(canvas, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    scale = ACTION_SCALE
    step  = args.step
    cumulative = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    camera = CameraController(index=args.index, fps=args.fps)
    camera.start()
    cam_thread = CameraThread(camera)
    cam_thread.start()

    robot = None
    if not args.no_robot:
        robot = TransferControl()
        print('Robot connected.')

    def move(axis: str, delta_mm: float):
        actual = delta_mm * scale
        cumulative[axis] += actual
        print(f'  {axis} {actual:+.3f} mm  (cumulative {cumulative[axis]:+.3f} mm)')
        if robot is not None:
            robot.move_axis_by(axis, actual)

    def home():
        if robot is None:
            return
        print('Homing...')
        for ax, total in cumulative.items():
            if total != 0.0:
                print(f'  return {ax} by {-total:+.3f} mm')
                robot.move_axis_by(ax, -total)
                cumulative[ax] = 0.0
        print('Homing complete.')

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, WIN_W, WIN_H)
    print('Ready. Use arrow keys / Q / E to move. ESC to home and quit.')

    try:
        while True:
            frame  = cam_thread.latest()
            canvas = draw_hud(frame, scale, cumulative, step)
            cv2.imshow(WIN_TITLE, canvas)

            key = cv2.waitKey(30) & 0xFFFFFF  # 30 ms poll → ~33 fps UI

            if key == 255 or key == -1:        # no key
                continue
            if key == 27:                      # ESC
                break

            key_low = key & 0xFF               # strip modifiers for letter keys

            if   key in (KEY_UP,    KEY_UP_L)    or key_low == ord('w'): move('y',  step)
            elif key in (KEY_DOWN,  KEY_DOWN_L)  or key_low == ord('s'): move('y', -step)
            elif key in (KEY_LEFT,  KEY_LEFT_L)  or key_low == ord('a'): move('x', -step)
            elif key in (KEY_RIGHT, KEY_RIGHT_L) or key_low == ord('d'): move('x',  step)
            elif key_low == ord('q'):                                      move('z',  step)
            elif key_low == ord('e'):                                      move('z', -step)
            elif key_low == ord('=') or key_low == ord('+'):
                scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
                print(f'  ACTION_SCALE → {scale:.1f}')
            elif key_low == ord('-'):
                scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
                print(f'  ACTION_SCALE → {scale:.1f}')

    except KeyboardInterrupt:
        pass
    finally:
        home()
        cam_thread.stop()
        cv2.destroyAllWindows()
        camera.stop()
        if robot is not None:
            robot.disconnect()
        print('Done.')


if __name__ == '__main__':
    main()
