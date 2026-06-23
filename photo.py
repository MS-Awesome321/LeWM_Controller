"""Take a photo using CameraController and save it to the current directory."""
import argparse
import time
from pathlib import Path
from hardware.camera_controller import CameraController


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out',   default=None,  help='Output filename (default: photo_<timestamp>.jpg)')
    p.add_argument('--index', type=int, default=0, help='Camera device index')
    p.add_argument('--fps',   type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.out or f'photo_{int(time.time())}.jpg'

    cam = CameraController(index=args.index, fps=args.fps)
    cam.start()
    try:
        path = cam.take_photo(out)
        print(f'Saved: {path}')
    finally:
        cam.stop()


if __name__ == '__main__':
    main()
