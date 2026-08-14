import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

import cv2

ACTION_SCALE = 0.1
DEBOUNCE = 20
CAM_FPS = 15


def parse_args():
    p = argparse.ArgumentParser(description='Manually jog the stage while viewing the live camera feed.')
    p.add_argument('--save', nargs='?', const='', default=None,
                    help='Save every captured frame to a video file in the current directory '
                         '(default filename: manual_control_<timestamp>.mp4)')
    return p.parse_args()


args = parse_args()

video_writer = None
save_path = (args.save or f"manual_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4") if args.save is not None else None

try:
    cam = CameraController(index=0, fps=CAM_FPS)
    cam.start()
    cv2.namedWindow('Manual Transfer Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Manual Transfer Control', 960, 540)

    arm = TransferControl(only_xyz=True)
    # for axis in ['x', 'y', 'z']:
    #     print(arm.get_kst_speed(axis))
    #     arm.set_kst_speed(axis, max_vel=10.0, accel=10000.0, min_vel=10.0)
    #     print(arm.get_kst_speed(axis))

    i = 0
    while True:
        frame  = cam.snap()

        if save_path is not None and video_writer is None:
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), CAM_FPS, (w, h))
            print(f'Saving captured frames to {save_path}')
        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Manual Transfer Control', frame)

        key = cv2.waitKey(1)

        if i >= 0:
            if key == 123 or key == 97: # a
                print('Left')
                i = -DEBOUNCE
                arm.jog_axis('x', '+')
            elif key == 124 or key == 100: # d
                print('Right')
                i = -DEBOUNCE
                arm.jog_axis('x', '-')
            elif key == 125 or key == 115: # s
                print('Down')
                i = -DEBOUNCE
                arm.jog_axis('y', '-')
            elif key == 126 or key == 119: # w
                print('Up')
                i = -DEBOUNCE
                arm.jog_axis('y', '+')
            elif key == 113: # q
                print('Raise')
                i = -DEBOUNCE
                arm.jog_axis('z', '+')
            elif key == 101: # e
                print('Lower')
                i = -DEBOUNCE
                arm.jog_axis('z', '-')
            elif key == 48:   # 0
                pos = arm.positions()
                fname = f"images/capture_{pos['x']}_{pos['y']}_{pos['z']}.png"
                cv2.imwrite(fname, frame)
                print(f'Saved {fname}')
            elif key == 112:  # p
                print(arm.positions())
            elif key == 27:   # ESC
                print("Stopping All")
                arm.stop_xyz()
                break
            else:
                arm.stop_xyz()
        
        i += 1

except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    arm.disconnect()
    if video_writer is not None:
        video_writer.release()
        print(f'Saved video to {save_path}')
    cv2.destroyAllWindows()
