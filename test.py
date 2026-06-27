import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hardware.camera_controller import CameraController
from hardware.transfer_control_controller import TransferControl

import cv2
import numpy as np

ACTION_SCALE = 0.1

cam = CameraController(index=0, fps=15)
cam.start()
cv2.namedWindow('Manual Transfer Control', cv2.WINDOW_NORMAL)

arm = TransferControl()

try:
    while True:
        frame  = cam.snap()
        cv2.imshow('Manual Transfer Control', frame)

        key = cv2.waitKey(1)

        if key == 123:
            print('Left')
        elif key == 124:
            print('Right')
        elif key == 125:
            print('Down')
        elif key == 126:
            print('Up')
        elif key == 27:   # ESC
            print("Stop: ", key)

            break
        else:
            print("Stop Jog: ", key)



except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    arm.disconnect()
    cv2.destroyAllWindows()
