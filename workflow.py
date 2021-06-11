import numpy as np
import cv2
from icecream import ic
from importlib import reload
from matplotlib import pyplot as plt

import calibration_tool
from stereocam import StereoCamera
import preprocessing
import algorithm
import traceback
import image_output

CALIBRATING = True
grab = False

def unhandled_key(key):
    def wrapper():
        print("unhandled keyboard input: ", key)
    return wrapper

def quit():
    raise KeyboardInterrupt

def grab_pcl():
    global grab
    grab = True

def reload_preprocessing():
    try:
        print("reloading preprocessing")
        reload(preprocessing)
        print("reloading preprocessing - successful")
    except Exception as e:
        print("reloading preprocessing - failed")
        ic(e)

def reload_algorithm():
    reload(image_output)
    try:
        print("reloading algorithm")
        reload(algorithm)
        print("reloading algorithm - successful")
    except Exception as e:
        print("reloading algorithm - failed")
        ic(e)

def capture_calibration_data():
    reload(calibration_tool)
    try:
        calibration_tool.capture_data_charuco()
    except Exception as e:
        ic(e)
        traceback.print_exc()
        _e = e

def run_calibration():
    reload(calibration_tool)
    try:
        calibration_tool.load_data_and_calibrate_charuco()
    except Exception as e:
        ic(e)
        traceback.print_exc()
        _e = e

def show_calibration():
    try:
        calibration_tool.print_stored_data_and_calibration()
    except Exception as e:
        ic(e)
        traceback.print_exc()
        _e = e

def main():
    global grab
    try:
        for frame in StereoCamera().frame_generator():
            l,r = preprocessing.run(frame, preprocessing.GRAY)
            d = algorithm.run(l, r, grab)
            grab = False
            key = cv2.waitKey(33)
            if key != -1:
                action = {
                    "q": quit,
                    "c": capture_calibration_data,
                    "r": run_calibration,
                    "a": reload_algorithm,
                    "s": show_calibration,
                    "g": grab_pcl,
                    "p": reload_preprocessing,
                }.get(chr(key), unhandled_key(key))
                action()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

