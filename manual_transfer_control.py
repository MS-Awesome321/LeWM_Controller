"""
Manual keyboard control for the nanochemistry transfer stage.

The camera liveview runs as a completely separate subprocess (liveview.py) so it
never pauses or freezes regardless of what the main process is doing.
Keyboard input is handled via pynput (no cv2 window needed in main).

Keys:
    Arrow Up    — move Y forward
    Arrow Down  — move Y backward
    Arrow Left  — move X left
    Arrow Right — move X right
    Q           — move Z up
    E           — move Z down
    =  /  +     — increase ACTION_SCALE by 0.1
    -           — decrease ACTION_SCALE by 0.1 (min 0.1)
    H           — home (return to start position)
    ESC / Ctrl-C — home and exit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pynput.keyboard import Events, Key

sys.path.insert(0, str(Path(__file__).parent))
from hardware.transfer_control_controller import TransferControl

# ── constants ─────────────────────────────────────────────────────────────────
STEP_MM      = 1.0
ACTION_SCALE = 0.1
SCALE_STEP   = 0.1
SCALE_MIN    = 0.1
SCALE_MAX    = 10.0


def parse_args():
    p = argparse.ArgumentParser(description='Manual transfer stage controller.')
    p.add_argument('--fps',      type=int,   default=15)
    p.add_argument('--index',    type=int,   default=0)
    p.add_argument('--step',     type=float, default=STEP_MM, help='Base step size (mm)')
    p.add_argument('--no_robot', action='store_true', help='No robot — keyboard + liveview only')
    return p.parse_args()


_SPECIAL = {
    Key.up:    'up',
    Key.down:  'down',
    Key.left:  'left',
    Key.right: 'right',
    Key.esc:   'esc',
}
_CHARS = {'q', 'e', 'h', '=', '+', '-'}


def parse_key(key) -> str | None:
    token = _SPECIAL.get(key)
    if token is None:
        try:
            c = key.char.lower() if key.char else None
            if c in _CHARS:
                token = '=' if c == '+' else c
        except AttributeError:
            pass
    return token


# ── helpers ───────────────────────────────────────────────────────────────────

def print_status(scale: float, step: float, cumulative: dict):
    eff = step * scale
    print(
        f'\r  scale={scale:.1f}  step={eff:.3f}mm  '
        f'x={cumulative["x"]:+.3f}  y={cumulative["y"]:+.3f}  z={cumulative["z"]:+.3f}  ',
        end='', flush=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    scale = ACTION_SCALE
    step  = args.step
    cumulative = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    # ── launch liveview subprocess ─────────────────────────────────────────────
    liveview_script = Path(__file__).parent / 'liveview.py'
    liveview_proc   = subprocess.Popen(
        [sys.executable, str(liveview_script),
         '--index', str(args.index), '--fps', str(args.fps)],
    )

    # ── connect robot ──────────────────────────────────────────────────────────
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
        for ax, total in cumulative.items():
            if total != 0.0:
                print(f'  return {ax} by {-total:+.3f} mm')
                robot.move_axis_by(ax, -total)
                cumulative[ax] = 0.0
        print('Homing complete.')

    print('Ready.  Arrows=XY  Q/E=Z  =/−=scale  H=home  ESC=quit')
    print_status(scale, step, cumulative)

    try:
        with Events() as events:
            for event in events:
                if not isinstance(event, Events.Press):
                    continue

                key = parse_key(event.key)
                if key is None:
                    continue

                if   key == 'up':    move('y',  step)
                elif key == 'down':  move('y', -step)
                elif key == 'left':  move('x', -step)
                elif key == 'right': move('x',  step)
                elif key == 'q':     move('z',  step)
                elif key == 'e':     move('z', -step)
                elif key == '=':
                    scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
                    print_status(scale, step, cumulative)
                elif key == '-':
                    scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
                    print_status(scale, step, cumulative)
                elif key == 'h':
                    home()
                    print_status(scale, step, cumulative)
                elif key == 'esc':
                    break

    except KeyboardInterrupt:
        pass
    finally:
        home()
        liveview_proc.terminate()
        liveview_proc.wait()
        if robot is not None:
            robot.disconnect()
        print('\nDone.')


if __name__ == '__main__':
    main()
