import cv2
from settings import *

GRAY = False
# GRAY = True

def split_h(frame):
    return frame[:, :FRAME_WIDTH, :], frame[:, FRAME_WIDTH:, :]

def run(frame, gray=True):
    if FLIP_CAMERA:
        frame = frame[::-1, ::-1, :]

    l,r = split_h(frame)

    if gray:
        l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

    return l, r
