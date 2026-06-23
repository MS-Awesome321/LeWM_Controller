"""
Standalone camera liveview window with optional CEM overlay.

When launched by cem_mpc.py the parent process pipes newline-delimited JSON to
stdin.  Each line carries the latest plan data and is rendered on top of the
camera frame.  When launched directly (or stdin is not a pipe) the window just
shows a plain camera feed.

JSON schema (one compact line per MPC step):
  {
    "step": int,
    "dist": float,
    "best": [[dx,dy,dz], ...],        # (horizon, 3) best action sequence
    "elite": [[[dx,dy,dz], ...], ...]  # (n_elite, horizon, 3) all elite seqs
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController

WIN_W, WIN_H = 800, 600
PX_PER_MM    = 8.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index',   type=int,   default=0)
    p.add_argument('--fps',     type=int,   default=15)
    p.add_argument('--w',       type=int,   default=WIN_W)
    p.add_argument('--h',       type=int,   default=WIN_H)
    return p.parse_args()


# ── stdin overlay reader ──────────────────────────────────────────────────────

class OverlayReader(threading.Thread):
    """Reads newline-delimited JSON from stdin and keeps the latest overlay."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock    = threading.Lock()
        self._overlay = None   # latest parsed dict, or None

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                with self._lock:
                    self._overlay = data
            except json.JSONDecodeError:
                pass

    def latest(self) -> dict | None:
        with self._lock:
            return self._overlay


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_overlay(canvas: np.ndarray, overlay: dict, w: int, h: int) -> None:
    cx, cy = w // 2, h // 2
    sx = PX_PER_MM
    sy = PX_PER_MM

    elite = np.array(overlay['elite'], dtype=np.float32)   # (E, H, 3)
    best  = np.array(overlay['best'],  dtype=np.float32)   # (H, 3)
    dist  = overlay['dist']
    step  = overlay['step']

    # dim grey elite trajectories
    for seq in elite:
        pts = [(cx, cy)]
        for dx, dy, _ in seq:
            pts.append((int(pts[-1][0] + dx * sx), int(pts[-1][1] - dy * sy)))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 2, (80, 80, 80), -1)

    # coloured best trajectory
    pts = [(cx, cy)]
    for dx, dy, _ in best:
        pts.append((int(pts[-1][0] + dx * sx), int(pts[-1][1] - dy * sy)))
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        dz  = float(best[i, 2])
        r   = max(0, min(255, int(128 + dz * 12)))
        g   = 220
        b_c = max(0, min(255, int(128 - dz * 12)))
        cv2.line(canvas, a, b, (b_c, g, r), 2, cv2.LINE_AA)
        cv2.circle(canvas, b, 3, (b_c, g, r), -1)
    cv2.circle(canvas, pts[0], 5, (0, 255, 255), -1)

    # text
    first = best[0]
    lines = [
        f'Step {step}',
        f'Dist to goal: {dist:.3f}',
        f'Next  x={first[0]:+.2f}  y={first[1]:+.2f}  z={first[2]:+.2f} mm',
    ]
    for i, text in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    w, h   = args.w, args.h

    cam = CameraController(index=args.index, fps=args.fps)
    cam.start()

    overlay_reader = OverlayReader()
    overlay_reader.start()

    cv2.namedWindow('CEM-MPC  |  elite plans', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CEM-MPC  |  elite plans', w, h)

    try:
        while True:
            frame  = cam.snap()                          # BGR uint8
            canvas = cv2.resize(frame, (w, h))

            overlay = overlay_reader.latest()
            if overlay is not None:
                draw_overlay(canvas, overlay, w, h)

            cv2.imshow('CEM-MPC  |  elite plans', canvas)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
