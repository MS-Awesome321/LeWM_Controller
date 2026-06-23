"""
Manual keyboard control for the nanochemistry transfer stage.

Hold a key to jog continuously; release to stop.
The camera liveview runs as a completely separate subprocess (liveview.py).

Keys:
    Arrow Up    — jog Y forward
    Arrow Down  — jog Y backward
    Arrow Left  — jog X left
    Arrow Right — jog X right
    Q           — jog Z up
    E           — jog Z down
    =  /  +     — increase ACTION_SCALE by 0.1  (tap)
    -           — decrease ACTION_SCALE by 0.1  (tap)
    H           — home  (tap)
    ESC / Ctrl-C — home and exit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from pynput.keyboard import Key, Listener

sys.path.insert(0, str(Path(__file__).parent))
from hardware.transfer_control_controller import TransferControl

# ── constants ─────────────────────────────────────────────────────────────────
STEP_MM      = 1.0    # mm sent per jog tick
JOG_HZ       = 10     # jog ticks per second while key held
ACTION_SCALE = 0.1
SCALE_STEP   = 0.1
SCALE_MIN    = 0.1
SCALE_MAX    = 10.0

_SPECIAL = {
    Key.up:    'up',
    Key.down:  'down',
    Key.left:  'left',
    Key.right: 'right',
    Key.esc:   'esc',
}
_JOG_KEYS  = {'up', 'down', 'left', 'right', 'q', 'e'}
_TAP_KEYS  = {'=', '-', 'h', 'esc'}


def parse_args():
    p = argparse.ArgumentParser(description='Manual transfer stage controller.')
    p.add_argument('--fps',      type=int,   default=15)
    p.add_argument('--index',    type=int,   default=0)
    p.add_argument('--step',     type=float, default=STEP_MM, help='Base step size per tick (mm)')
    p.add_argument('--hz',       type=float, default=JOG_HZ,  help='Jog ticks per second')
    p.add_argument('--no_robot', action='store_true')
    return p.parse_args()


def key_to_token(key) -> str | None:
    token = _SPECIAL.get(key)
    if token is None:
        try:
            c = key.char.lower() if key.char else None
            if c in ('q', 'e', 'h', '=', '+', '-'):
                token = '=' if c == '+' else c
        except AttributeError:
            pass
    return token


def print_status(scale: float, step: float, cumulative: dict):
    eff = step * scale
    print(
        f'\r  scale={scale:.1f}  step={eff:.3f}mm  '
        f'x={cumulative["x"]:+.3f}  y={cumulative["y"]:+.3f}  z={cumulative["z"]:+.3f}  ',
        end='', flush=True,
    )


def main():
    args  = parse_args()
    scale = ACTION_SCALE
    step  = args.step
    hz    = args.hz
    cumulative = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    stop_flag  = threading.Event()

    # ── liveview subprocess ────────────────────────────────────────────────────
    liveview_script = Path(__file__).parent / 'liveview.py'
    liveview_proc   = subprocess.Popen(
        [sys.executable, str(liveview_script),
         '--index', str(args.index), '--fps', str(args.fps)],
    )

    # ── robot ──────────────────────────────────────────────────────────────────
    robot = None
    if not args.no_robot:
        robot = TransferControl()
        print('Robot connected.')

    def move(axis: str, delta_mm: float):
        actual = delta_mm * scale
        cumulative[axis] += actual
        if robot is not None:
            robot.move_axis_by(axis, actual)
        print_status(scale, step, cumulative)

    def home():
        if robot is None:
            return
        print('\nHoming...')
        for ax, total in list(cumulative.items()):
            if total != 0.0:
                print(f'  return {ax} by {-total:+.3f} mm')
                robot.move_axis_by(ax, -total)
                cumulative[ax] = 0.0
        print('Homing complete.')

    # ── held-key tracking ─────────────────────────────────────────────────────
    held = set()   # tokens currently held down
    held_lock = threading.Lock()

    def on_press(key):
        nonlocal scale
        token = key_to_token(key)
        if token is None:
            return

        if token in _JOG_KEYS:
            with held_lock:
                held.add(token)
        elif token == '=':
            scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
            print_status(scale, step, cumulative)
        elif token == '-':
            scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
            print_status(scale, step, cumulative)
        elif token == 'h':
            home()
            print_status(scale, step, cumulative)
        elif token == 'esc':
            stop_flag.set()
            return False   # stop listener

    def on_release(key):
        token = key_to_token(key)
        if token in _JOG_KEYS:
            with held_lock:
                held.discard(token)

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print('Ready.  Hold Arrows/QE to jog.  =/- scale.  H=home.  ESC=quit.')
    print_status(scale, step, cumulative)

    interval = 1.0 / hz
    try:
        while not stop_flag.is_set():
            with held_lock:
                active = set(held)

            if active:
                for token in active:
                    if   token == 'up':    move('y',  step)
                    elif token == 'down':  move('y', -step)
                    elif token == 'left':  move('x', -step)
                    elif token == 'right': move('x',  step)
                    elif token == 'q':     move('z',  step)
                    elif token == 'e':     move('z', -step)
                time.sleep(interval)
            else:
                time.sleep(0.01)   # idle — stay responsive without burning CPU

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        home()
        liveview_proc.terminate()
        liveview_proc.wait()
        if robot is not None:
            robot.disconnect()
        print('\nDone.')


if __name__ == '__main__':
    main()
