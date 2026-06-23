"""
Manual keyboard control for the nanochemistry transfer stage with live camera view.

The camera runs in a completely separate process and writes frames into a shared
memory block.  The main process reads from shared memory and handles the cv2
window + keyboard + robot moves — no GIL contention, no thread scheduling jitter.

Keys:
    Arrow Up    — move Y forward
    Arrow Down  — move Y backward
    Arrow Left  — move X left
    Arrow Right — move X right
    Q           — move Z up
    E           — move Z down
    =  /  +     — increase ACTION_SCALE by 0.1
    -           — decrease ACTION_SCALE by 0.1 (min 0.1)
    ESC / Ctrl-C — home and exit
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Process, Value, Event
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── constants ─────────────────────────────────────────────────────────────────
STEP_MM      = 1.0
ACTION_SCALE = 0.1
SCALE_STEP   = 0.1
SCALE_MIN    = 0.1
SCALE_MAX    = 10.0
WIN_TITLE    = 'Manual Transfer Control'
WIN_W, WIN_H = 800, 600

# Shared-memory frame shape: camera native size (resized when displaying)
CAM_H, CAM_W = 480, 640   # adjust if your camera uses a different resolution

# cv2 arrow key codes — Windows extended codes and Linux/macOS single-byte codes
KEY_UP     = 2490368;  KEY_UP_L    = 82
KEY_DOWN   = 2621440;  KEY_DOWN_L  = 84
KEY_LEFT   = 2424832;  KEY_LEFT_L  = 81
KEY_RIGHT  = 2555904;  KEY_RIGHT_L = 83


# ── camera process ────────────────────────────────────────────────────────────

def _camera_process(shm_name: str, frame_counter,
                    stop_event, cam_index: int, cam_fps: int):
    """
    Runs in a separate process.  Continuously snaps frames and writes them into
    the named SharedMemory block, then increments frame_counter so the reader
    knows a new frame is available.
    """
    from hardware.camera_controller import CameraController

    buf = SharedMemory(name=shm_name)
    arr = np.ndarray((CAM_H, CAM_W, 3), dtype=np.uint8, buffer=buf.buf)

    cam = CameraController(index=cam_index, fps=cam_fps)
    cam.start()
    try:
        while not stop_event.is_set():
            try:
                bgr = cam.snap()                     # (H, W, 3) BGR uint8
                if bgr.shape[:2] != (CAM_H, CAM_W):
                    bgr = cv2.resize(bgr, (CAM_W, CAM_H))
                np.copyto(arr, bgr)
                with frame_counter.get_lock():
                    frame_counter.value += 1
            except Exception:
                pass
    finally:
        cam.stop()
        buf.close()


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Manual transfer stage controller.')
    p.add_argument('--fps',      type=int,   default=15)
    p.add_argument('--index',    type=int,   default=0)
    p.add_argument('--step',     type=float, default=STEP_MM, help='Base step size (mm)')
    p.add_argument('--no_robot', action='store_true', help='Camera only, no robot')
    return p.parse_args()


def draw_hud(frame_bgr: np.ndarray, scale: float,
             cumulative: dict, step_mm: float) -> np.ndarray:
    canvas = cv2.resize(frame_bgr, (WIN_W, WIN_H))
    effective = step_mm * scale
    lines = [
        f'ACTION_SCALE: {scale:.1f}   step = {effective:.3f} mm',
        f'Cumulative:  x={cumulative["x"]:+.3f}  y={cumulative["y"]:+.3f}  z={cumulative["z"]:+.3f} mm',
        'Arrows / WASD = XY   Q/E = Z   =/- = scale   ESC = home + quit',
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

    # ── shared memory for camera frames ───────────────────────────────────────
    frame_bytes   = CAM_H * CAM_W * 3
    shm           = SharedMemory(create=True, size=frame_bytes)
    shm_arr       = np.ndarray((CAM_H, CAM_W, 3), dtype=np.uint8, buffer=shm.buf)
    frame_counter = Value('i', 0)
    stop_event    = Event()

    cam_proc = Process(
        target=_camera_process,
        args=(shm.name, frame_counter, stop_event, args.index, args.fps),
        daemon=True,
    )
    cam_proc.start()

    # ── robot ──────────────────────────────────────────────────────────────────
    robot = None
    if not args.no_robot:
        from hardware.transfer_control_controller import TransferControl
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

    # ── UI loop ───────────────────────────────────────────────────────────────
    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, WIN_W, WIN_H)
    print('Ready.  Arrow keys / WASD = XY,  Q/E = Z,  =/- = scale,  ESC = quit.')

    last_seen = -1
    try:
        while True:
            # read latest frame if the camera process wrote a new one
            with frame_counter.get_lock():
                current = frame_counter.value
            if current != last_seen:
                frame    = shm_arr.copy()   # snapshot — camera process may overwrite
                last_seen = current

            canvas = draw_hud(frame, scale, cumulative, step)
            cv2.imshow(WIN_TITLE, canvas)

            key = cv2.waitKey(30) & 0xFFFFFF

            if key in (255, 0xFFFFFF):
                continue
            if key == 27:                    # ESC
                break

            kl = key & 0xFF                  # single-byte value for letter keys

            if   key in (KEY_UP,    KEY_UP_L)    or kl == ord('w'): move('y',  step)
            elif key in (KEY_DOWN,  KEY_DOWN_L)  or kl == ord('s'): move('y', -step)
            elif key in (KEY_LEFT,  KEY_LEFT_L)  or kl == ord('a'): move('x', -step)
            elif key in (KEY_RIGHT, KEY_RIGHT_L) or kl == ord('d'): move('x',  step)
            elif kl == ord('q'):                                      move('z',  step)
            elif kl == ord('e'):                                      move('z', -step)
            elif kl in (ord('='), ord('+')):
                scale = min(SCALE_MAX, round(scale + SCALE_STEP, 10))
                print(f'  ACTION_SCALE → {scale:.1f}')
            elif kl == ord('-'):
                scale = max(SCALE_MIN, round(scale - SCALE_STEP, 10))
                print(f'  ACTION_SCALE → {scale:.1f}')

    except KeyboardInterrupt:
        pass
    finally:
        home()
        stop_event.set()
        cam_proc.join(timeout=3)
        cv2.destroyAllWindows()
        shm.close()
        shm.unlink()
        if robot is not None:
            robot.disconnect()
        print('Done.')


if __name__ == '__main__':
    main()
