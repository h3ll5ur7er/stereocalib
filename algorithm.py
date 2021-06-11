import stereo_algo
import numpy as np
import cv2
from importlib import reload
from icecream import ic
import traceback

_e = None
def run(l, r, grab):
    global _e
    try:
        try:
            reload(stereo_algo)
        except:
            pass
        d = stereo_algo.calculate_disparity(l, r, grab)
        return d
    except Exception as e:
        if not _e or str(e) != str(_e):
            ic(e)
            traceback.print_exc()
            _e = e
        return np.zeros_like(l)