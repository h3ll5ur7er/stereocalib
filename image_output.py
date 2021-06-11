import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from settings import *


def put_name(img, name):
    img = np.ascontiguousarray(img)
    return cv2.putText(img, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 0xff), 2)

def normalize(img):
    return cv2.normalize(img.copy(), None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)

def edges(img):
    return cv2.Sobel(img.copy(), 5, 1, 1)

def scale(img):
    return cv2.resize(img, tuple(np.array(img.shape[:2][::-1])//DOWN_SCALE))

def to_bgr(*images):
    bgr = []
    for image in images:
        if len(image.shape) == 2:
            try:
                image = np.ascontiguousarray(image)
                image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(e)
        bgr.append(image)
    return bgr

def group(*images):
    return np.vstack(to_bgr(*images))

def concat(*images):
    return np.hstack(to_bgr(*images))

def show_unscaled(img, title):
    cv2.imshow(title, img)

def show(img, title):
    cv2.imshow(title, scale(img))

def show_normalized(img, title):
    show(normalize(img), title)

def show_edges(img, title):
    show(edges(img), title)

def show_normalized_edges(img, title):
    show(normalize(edges(img)), title)
