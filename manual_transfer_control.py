"""
Manual keyboard control for the nanochemistry transfer stage.

Hold a key to jog continuously; release to stop immediately.
Camera feed runs in the main process via a cv2 window.

Keys:
    Arrow Up    — jog Y forward
    Arrow Down  — jog Y backward
    Arrow Left  — jog X left
    Arrow Right — jog X right
    Q           — jog Z up
    E           — jog Z down
    =  /  +     — increase ACTION_SCALE by 0.1
    -           — decrease ACTION_SCALE by 0.1
    H           — home (return to start position)
    ESC         — home and exit
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

# ── constants ─────────────────────────────────────────────────────────────────
STEP_MM      = 0.1    # mm per jog command
ACTION_SCALE = 0.1
SCALE_STEP   = 0.1
SCALE_MIN    = 0.1
SCALE_MAX    = 10.0
WIN_W, WIN_H = 800, 600

# cv2 arrow key codes
KEY_UP    = 2490368; KEY_UP_L    = 82
KEY_DOWN  = 2621440; KEY_DOWN_L  = 84
KEY_LEFT  = 2424832; KEY_LEFT_L  = 81
KEY_RIGHT = 2555904; KEY_RIGHT_L = 83


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fps',      type=int,   default=15)
    p.add_argument('--index',    type=int,   default=0)
    p.add_argument('--step',     type=float, default=STEP_MM)
    p.add_argument('--no_robot', action='store_true')
    return p.parse_args()


def key_to_jog(key: int) -> tuple[str, float] | None:
    """Map a cv2 key code to (axis, sign). Returns None for non-jog keys."""
    kl = key & 0xFF
    if key in (KEY_UP,    KEY_UP_L):    return ('y',  1.0)
    if key in (KEY_DOWN,  KEY_DOWN_L):  return ('y', -1.0)
    if key in (KEY_LEFT,  KEY_LEFT_L):  return ('x', -1.0)
    if key in (KEY_RIGHT, KEY_RIGHT_L): return ('x',  1.0)
    if kl == ord('q'): return ('z',  1.0)
    if kl == ord('e'): return ('z', -1.0)
    return None


def draw_hud(frame: np.ndarray, scale: float, step: float,
             cumulative: dict, jogging: str | None) -> np.ndarray:
    canvas = cv2.resize(frame, (WIN_W, WIN_H))
    eff    = step * scale
    status = f'JOGGING {jogging}' if jogging else 'idle'
    lines  = [
        f'ACTION_SCALE: {scale:.1f}   step = {eff:.3f} mm   [{status}]',
        f'x={cumulative["x"]:+.3f}  y={cumulative["y"]:+.3f}  z={cumulative["z"]:+.3f} mm',
        'Arrows/QE=jog   =/- scale   H=home   ESC=quit',
    ]
    for i, text in enumerate(lines):
        y = WIN_H - 20 - (len(lines) - 1 - i) * 22
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    args  = parse_args()
    scale = ACTION_SCALE
    step  = args.step
    cumulative = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    cam = CameraController(index=args.index, fps=args.fps)
    cam.start()

    robot = None
    if not args.no_robot:
        robot = TransferControl()
        print('Robot connected.')

    # ── jog thread ────────────────────────────────────────────────────────────
    # One persistent thread sends move commands while a stop event is clear.
    # Replacing _jog_stop stops the current jog instantly.
    _jog_stop  = threading.Event()
    _jog_stop.set()   # start in stopped state
    _jog_token = [None]   # ['x+'] etc. — just for HUD

    def _jog_worker(axis: str, delta: float, stop: threading.Event):
        while not stop.is_set():
            if robot is not None:
                robot.move_axis_by(axis, delta)
            cumulative[axis] += delta

    _jog_thread: threading.Thread | None = None

    def start_jog(axis: str, sign: float):
        nonlocal _jog_stop, _jog_thread
        delta = sign * step * scale
        # already jogging the same axis+direction — nothing to change
        if _jog_token[0] == (axis, sign):
            return
        stop_jog()
        _jog_stop  = threading.Event()
        _jog_token[0] = (axis, sign)
        _jog_thread = threading.Thread(
            target=_jog_worker, args=(axis, delta, _jog_stop), daemon=True
        )
        _jog_thread.start()

    def stop_jog():
        nonlocal _jog_thread
        _jog_stop.set()
        if _jog_thread is not None:
            _jog_thread.join()
            _jog_thread = None
        _jog_token[0] = None

    def home():
        stop_jog()
        if robot is None:
            return
        print('\nHoming...')
        for ax, total in list(cumulative.items()):
            if total != 0.0:
                print(f'  return {ax} by {-total:+.3f} mm')
                robot.move_axis_by(ax, -total)
                cumulative[ax] = 0.0
        print('Homing complete.')

    # ── main cv2 loop ─────────────────────────────────────────────────────────
    cv2.namedWindow('Manual Transfer Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Manual Transfer Control', WIN_W, WIN_H)

    try:
        while True:
            frame  = cam.snap()
            canvas = draw_hud(frame, scale, step, cumulative, _jog_token[0])
            cv2.imshow('Manual Transfer Control', canvas)

            key = cv2.waitKey(1) & 0xFFFFFF
            if key in (0xFFFFFF, 0xFF):
                key = -1

            jog = key_to_jog(key) if key != -1 else None

            if jog:
                start_jog(*jog)
            else:
                if not _jog_stop.is_set():
                    stop_jog()

            if key == 27:   # ESC
                break

            kl = key & 0xFF if key != -1 else -1
            if kl == ord('=') or kl == ord('+'):
                scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
            elif kl == ord('-'):
                scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
            elif kl == ord('h'):
                home()

    except KeyboardInterrupt:
        pass
    finally:
        home()
        cv2.destroyAllWindows()
        cam.stop()
        if robot is not None:
            robot.disconnect()
        print('Done.')


if __name__ == '__main__':
    main()
