import cv2
from utils import Singleton 
from settings import CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT, CAMERA_ID

class StereoCamera(metaclass=Singleton):
    def __init__(self):
        self.cap = cap = cv2.VideoCapture(CAMERA_ID,cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)

    def frame_generator(self):
        running = True
        while running:
            running, frame = self.cap.read()
            if running:
                yield frame