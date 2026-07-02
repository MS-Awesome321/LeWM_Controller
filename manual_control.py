import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

import cv2
import numpy as np

ACTION_SCALE = 0.1
DEBOUNCE = 20

try:
    cam = CameraController(index=0, fps=15)
    cam.start()
    cv2.namedWindow('Manual Transfer Control', cv2.WINDOW_NORMAL)

    arm = TransferControl(only_xyz=True)
    # for axis in ['x', 'y', 'z']:
    #     print(arm.get_kst_speed(axis))
    #     arm.set_kst_speed(axis, max_vel=10.0, accel=10000.0, min_vel=10.0)
    #     print(arm.get_kst_speed(axis))

    i = 0
    while True:
        frame  = cam.snap()
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
                arm.jog_axis('y', '+')
            elif key == 126 or key == 119: # w
                print('Up')
                i = -DEBOUNCE
                arm.jog_axis('y', '-')
            elif key == 113: # q
                print('Raise')
                i = -DEBOUNCE
                arm.jog_axis('z', '+')
            elif key == 101: # e
                print('Lower')
                i = -DEBOUNCE
                arm.jog_axis('z', '-')
            elif key == 48:   # 0
                fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame)
                print(f'Saved {fname}')
            elif key == 27:   # ESC
                print("Stopping All: ", key)
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
    cv2.destroyAllWindows()
