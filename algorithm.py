import stereo_algo
import numpy as np
import cv2
from importlib import reload
from icecream import ic
import traceback
import settings
from status_service import StatusService

_e = None
def run(l, r, grab):
    StatusService().status = "idle"
    global _e
    try:
        try:
            reload(stereo_algo)
        except:
            pass
        status_screen = np.zeros((settings.CAMERA_RESOLUTION_HEIGHT, settings.FRAME_WIDTH))
        for i, line in enumerate(StatusService().usage):
            status_screen = cv2.putText(status_screen, line, (100, 40*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, 0xff, 2)
        status_screen = cv2.putText(status_screen, StatusService().status, (100, 50*(len(StatusService().usage)+1)), cv2.FONT_HERSHEY_SIMPLEX, 2, 0xff, 2)
        cv2.imshow("status", status_screen)
        d = stereo_algo.calculate_disparity(l, r, grab)
        return d
    except Exception as e:
        StatusService().status = "error"
        if not _e or str(e) != str(_e):
            ic(e)
            traceback.print_exc()
            _e = e
        return np.zeros_like(l)