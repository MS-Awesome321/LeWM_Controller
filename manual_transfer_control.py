"""
Manual keyboard control for the nanochemistry transfer stage.

Hold a key to jog; release to stop. Uses the KST201 non-blocking MoveJog /
StopImmediate API so the cv2 loop never blocks.

Keys (arrows or WASD):
    Up / W      — jog Y forward
    Down / S    — jog Y backward
    Left / A    — jog X left
    Right / D   — jog X right
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
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

# ── constants ─────────────────────────────────────────────────────────────────
ACTION_SCALE = 0.1
SCALE_STEP   = 0.1
SCALE_MIN    = 0.1
SCALE_MAX    = 10.0
WIN_W, WIN_H = 800, 600

KEY_UP    = 2490368;  KEY_UP_L    = 82
KEY_DOWN  = 2621440;  KEY_DOWN_L  = 84
KEY_LEFT  = 2424832;  KEY_LEFT_L  = 81
KEY_RIGHT = 2555904;  KEY_RIGHT_L = 83


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fps',      type=int,   default=15)
    p.add_argument('--index',    type=int,   default=0)
    p.add_argument('--no_robot', action='store_true')
    return p.parse_args()


def key_to_jog(key: int) -> tuple[str, str] | None:
    """Map cv2 key → (axis, 'forward'|'backward'), or None."""
    kl = key & 0xFF
    if key in (KEY_UP,    KEY_UP_L)    or kl == ord('w'): return ('y', 'forward')
    if key in (KEY_DOWN,  KEY_DOWN_L)  or kl == ord('s'): return ('y', 'backward')
    if key in (KEY_LEFT,  KEY_LEFT_L)  or kl == ord('a'): return ('x', 'backward')
    if key in (KEY_RIGHT, KEY_RIGHT_L) or kl == ord('d'): return ('x', 'forward')
    if kl == ord('q'): return ('z', 'forward')
    if kl == ord('e'): return ('z', 'backward')
    return None


def draw_hud(frame: np.ndarray, scale: float, jogging: tuple | None) -> np.ndarray:
    canvas = cv2.resize(frame, (WIN_W, WIN_H))
    status = f'JOGGING {jogging[0]} {jogging[1]}' if jogging else 'idle'
    lines  = [
        f'ACTION_SCALE: {scale:.1f}   [{status}]',
        'Arrows/WASD=XY   Q/E=Z   =/- scale   H=home   ESC=quit',
    ]
    for i, text in enumerate(lines):
        y = WIN_H - 12 - (len(lines) - 1 - i) * 22
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    args  = parse_args()
    scale = ACTION_SCALE
    cumulative = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    cam = CameraController(index=args.index, fps=args.fps)
    cam.start()

    robot = None
    if not args.no_robot:
        robot = TransferControl()
        print('Robot connected.')

    active_jog: tuple[str, str] | None = None   # (axis, direction) currently jogging

    def start_jog(axis: str, direction: str):
        nonlocal active_jog
        if active_jog == (axis, direction):
            return
        if active_jog is not None:
            if robot is not None:
                robot.stop_axis(active_jog[0])
        active_jog = (axis, direction)
        if robot is not None:
            robot.start_jog_axis(axis, direction)

    def stop_jog():
        nonlocal active_jog
        if active_jog is not None:
            if robot is not None:
                robot.stop_axis(active_jog[0])
            active_jog = None

    def home():
        stop_jog()
        if robot is None:
            return
        print('\nHoming...')
        pos = robot.positions()
        for ax in ('x', 'y', 'z'):
            delta = -pos[ax]
            if abs(delta) > 1e-4:
                print(f'  return {ax} by {delta:+.3f} mm')
                robot.move_axis_by(ax, delta)
        print('Homing complete.')

    cv2.namedWindow('Manual Transfer Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Manual Transfer Control', WIN_W, WIN_H)

    try:
        while True:
            frame  = cam.snap()
            canvas = draw_hud(frame, scale, active_jog)
            cv2.imshow('Manual Transfer Control', canvas)

            key = cv2.waitKey(1) & 0xFFFFFF
            if key in (0xFFFFFF, 0xFF):
                key = -1

            if key == 27:   # ESC
                break

            jog = key_to_jog(key) if key != -1 else None

            if jog:
                start_jog(*jog)
            else:
                stop_jog()

            kl = key & 0xFF if key != -1 else -1
            if kl in (ord('='), ord('+')):
                scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
            elif kl == ord('-'):
                scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
            elif kl == ord('h'):
                home()

    except KeyboardInterrupt:
        pass
    finally:
        stop_jog()
        home()
        cv2.destroyAllWindows()
        cam.stop()
        if robot is not None:
            robot.disconnect()
        print('Done.')


if __name__ == '__main__':
    main()
