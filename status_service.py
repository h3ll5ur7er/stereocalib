import cv2
from utils import Singleton

class StatusService(metaclass=Singleton):
    def __init__(self):
        self.usage = [
            'General:',
            '"q": quit',
            '"c": capture_calibration_data',
            '"r": run_calibration',
            '"a": reload_algorithm',
            '"s": show_calibration',
            '"g": grab_pcl',
            '"p": reload_preprocessing',
            '',
            'Calibration:',
            '"ESCAPE": abort',
            '"<any>": confirm',
        ]
        self._status = "idle"
    @property
    def status(self):
        return self._status
    @status.setter
    def status(self, value):
        self._status = value
        cv2.waitKey(1)
