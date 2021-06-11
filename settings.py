import cv2

# Common

CAMERA_ID = 0
FLIP_CAMERA = True

CAMERA_RESOLUTION_WIDTH = 2560
CAMERA_RESOLUTION_HEIGHT = 720
FRAME_WIDTH = CAMERA_RESOLUTION_WIDTH // 2


# Calibration

CHECKERBOARD = (7,9)
CHECKERBOARD_SQUARE_SIZE_MM = 20
CHECKERBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

BLURRYNESS_THRESHOLD = 1000.0

EXPORT_DATA_FILENAME = "calibrationData.pkl"
IMPORT_DATA_FILENAME = "calibrationData.pkl"
EXPORT_CALIBRATION_FILENAME = "calibration.pkl"
IMPORT_CALIBRATION_FILENAME = "calibration.pkl"

EXPORT_DATA_FILENAME_CHARUCO = "calibrationData_charuco.pkl"
IMPORT_DATA_FILENAME_CHARUCO = "calibrationData_charuco.pkl"
EXPORT_CALIBRATION_FILENAME_CHARUCO = "calibration_charuco.pkl"
IMPORT_CALIBRATION_FILENAME_CHARUCO = "calibration_charuco.pkl"

SUB_PIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ALGORITHM

ACTIVE_CALIBRATION_FILENAME = "calibration.pkl"
ACTIVE_CALIBRATION_FILENAME = "calibration_charuco.pkl"
ACTIVE_CALIBRATION_FILENAME = "calibrations/calibration.pkl"

DEBUG = False
DOWN_SCALE = 4

# matcher = "bm"
MATCHER = "sgbm"

MAX_DISPARITY = 1000

BM_SETTINGS = {
    "numDisparities": 16,
    "blockSize": 5
}

SGBM_SETTINGS = {
    "minDisparity": 0,
    "numDisparities": 128,
    "blockSize": 8,
    "P1":24*64,
    "P2":96*64,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 10,
    "speckleWindowSize": 10,
    "speckleRange": 8,
    "mode": cv2.StereoSGBM_MODE_SGBM_3WAY,
}

WLS_LAMBDA = 8000.0
WLS_SIGMA_COLOR = 1.5


ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
CHARUCO_SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(
    squaresX=9,
    squaresY=7,
    squareLength=20.,
    markerLength=15.,
    dictionary=ARUCO_DICT)