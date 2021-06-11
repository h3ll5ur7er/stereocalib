#!/usr/bin/env python

import cv2
import sys
import numpy as np
import glob
from stereo_calibration import StereoCalibration, StereoCalibrationData
import pickle
import uuid

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)
FLIP_CAMERA = True

CAMERA_ID = 1
CAMERA_RESOLUTION_WIDTH = 2560
CAMERA_RESOLUTION_HEIGHT = 720

FRAME_WIDTH = CAMERA_RESOLUTION_WIDTH // 2

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,9)
CHECKERBOARD_SQUARE_SIZE_MM = 20
CHECKERBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

BLURRYNESS_THRESHOLD = 1000.0

EXPORT_DATA_FILENAME = "calibrationData.pkl"
IMPORT_DATA_FILENAME = "calibrationData.pkl"
EXPORT_CALIBRATION_FILENAME = "calibration.pkl"
IMPORT_CALIBRATION_FILENAME = "calibration.pkl"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def round(f):
    return int(f+0.5)

def get_checkerboard_aabb(corners):
    max_x, max_y = float("-inf"), float("-inf")
    min_x, min_y = float("inf"),  float("inf")
    for x,y in corners[:,0,:]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    w = max_x - min_x
    h = max_y - min_y
    return round(min_x), round(min_y), round(w), round(h)

def calculate_blurryness(img, x,y,w,h, lbl=""):
    roi = img[y:y+h, x:x+w]
    cv2.imshow(lbl, img)
    cv2.imshow("roi "+lbl,roi)
    blurryness = cv2.Laplacian(roi, cv2.CV_64F).var()
    return blurryness

def generate_object_points(scale):
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp*=scale
    return objp

def setup_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)
    return cap

def draw_rect(img, x,y,w,h, color, mode="add"):
    new_img = img.copy()
    color = np.array(color, dtype=np.uint8)
    if mode == "add":
        new_img[y:y+h, x:x+w] += color
    elif mode == "set":
        new_img[y:y+h, x:x+w] = color
    return new_img

def capture_calibration_data(save_images=False):
    calibration_data = StereoCalibrationData(uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex)
    objp = generate_object_points(CHECKERBOARD_SQUARE_SIZE_MM)
    cap = setup_camera(CAMERA_ID)
    l_viz = None
    r_viz = None
    capturing = True
    while capturing:
        capturing, img = cap.read()
        if not capturing:
            continue

        if FLIP_CAMERA:
            img = img[::-1, ::-1, :]
        l_img, r_img = img[:, :FRAME_WIDTH, :], img[:, FRAME_WIDTH:, :]
        if l_viz is None:
            l_viz = np.zeros_like(l_img)
        if r_viz is None:
            r_viz = np.zeros_like(r_img)

        l_gray = cv2.cvtColor(src=l_img, code=cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(src=r_img, code=cv2.COLOR_BGR2GRAY)

        l_found, l_corners = cv2.findChessboardCorners(
            l_gray,
            CHECKERBOARD,
            flags=CHECKERBOARD_FLAGS)
        r_found, r_corners = cv2.findChessboardCorners(
            r_gray,
            CHECKERBOARD,
            flags=CHECKERBOARD_FLAGS)

        if l_found and r_found:
            l_corners2 = cv2.cornerSubPix(l_gray, l_corners, (11,11),(-1,-1), criteria)
            r_corners2 = cv2.cornerSubPix(r_gray, r_corners, (11,11),(-1,-1), criteria)
            l_roi = get_checkerboard_aabb(l_corners2)
            l_blur = calculate_blurryness(l_img, *l_roi, lbl="l")
            r_roi = get_checkerboard_aabb(r_corners2)
            r_blur = calculate_blurryness(r_img, *r_roi, lbl="r")

            if l_blur < BLURRYNESS_THRESHOLD or r_blur < BLURRYNESS_THRESHOLD:
                continue
            l_viz = draw_rect(l_viz, *l_roi, (0, 0, 64))
            r_viz = draw_rect(r_viz, *r_roi, (0, 0, 64))

            calibration_data.left.frames.append(l_img)
            calibration_data.right.frames.append(r_img)

            calibration_data.object_points.append(objp)
            calibration_data.left.object_points.append(objp)
            calibration_data.right.object_points.append(objp)

            calibration_data.left.image_points.append(l_corners2)
            calibration_data.right.image_points.append(r_corners2)

            # Draw and display the corners
            l_corners_img = cv2.drawChessboardCorners(l_gray, CHECKERBOARD, l_corners2, l_found)
            r_corners_img = cv2.drawChessboardCorners(r_gray, CHECKERBOARD, r_corners2, r_found)

            cv2.imshow('l_img',l_corners_img)
            cv2.imshow('r_img',r_corners_img)
            cv2.imshow('l_viz',l_viz)
            cv2.imshow('r_viz',r_viz)
            if save_images:
                cv2.imwrite(f'images/{calibration_data.uid}/l_{len(calibration_data.left.image_points):03}.png',l_img)
                cv2.imwrite(f'images/{calibration_data.left.uid}/l_{len(calibration_data.left.image_points):03}.png',l_img)
                cv2.imwrite(f'images/{calibration_data.uid}/r_{len(calibration_data.left.image_points):03}.png',r_img)
                cv2.imwrite(f'images/{calibration_data.right.uid}/r_{len(calibration_data.left.image_points):03}.png',r_img)
            key = cv2.waitKey(1000)
        else:
            cv2.imshow('l_img',l_img)
            cv2.imshow('r_img',r_img)
            key = cv2.waitKey(33)
        if len(calibration_data.left.image_points)> 1000 or key != -1:
            if chr(key) == "q":
                exit()
            capturing = False

    calibration_data.image_size = l_gray.shape[::-1]
    calibration_data.left.image_size = l_gray.shape[::-1]
    calibration_data.right.image_size = r_gray.shape[::-1]

    cv2.destroyAllWindows()
    return calibration_data

def load_calibration_data(glob_pattern='./images/l_*.png', right_replacement=("l_", "r_")):
    calibration_data = StereoCalibrationData(uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex)
    objp = generate_object_points(CHECKERBOARD_SQUARE_SIZE_MM)

    images = glob.glob('./images/l_*.png')
    for fname in images:
        rname = fname.replace(*right_replacement)
        l_img = cv2.imread(fname)
        r_img = cv2.imread(rname)

        calibration_data.left.frames.append(l_img)
        calibration_data.right.frames.append(r_img)

        l_gray = cv2.cvtColor(src=l_img, code=cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(src=r_img, code=cv2.COLOR_BGR2GRAY)

        l_found, l_corners = cv2.findChessboardCorners(
            l_gray,
            CHECKERBOARD,
            flags=CHECKERBOARD_FLAGS)
        r_found, r_corners = cv2.findChessboardCorners(
            r_gray,
            CHECKERBOARD,
            flags=CHECKERBOARD_FLAGS)

        if l_found and r_found:
            calibration_data.object_points.append(objp)
            calibration_data.left.object_points.append(objp)
            calibration_data.right.object_points.append(objp)

            l_corners2 = cv2.cornerSubPix(l_gray, l_corners, (11,11),(-1,-1), criteria)
            r_corners2 = cv2.cornerSubPix(r_gray, r_corners, (11,11),(-1,-1), criteria)

            calibration_data.left.image_points.append(l_corners2)
            calibration_data.right.image_points.append(r_corners2)

            # Draw and display the corners
            l_corners_img = cv2.drawChessboardCorners(l_gray, CHECKERBOARD, l_corners2, l_found)
            r_corners_img = cv2.drawChessboardCorners(r_gray, CHECKERBOARD, r_corners2, r_found)
            cv2.imshow('l_corners_img',l_corners_img)
            cv2.imshow('r_corners_img',r_corners_img)
        cv2.imshow('l_img',l_img)
        cv2.imshow('r_img',r_img)
        key = cv2.waitKey(33)
        if len(calibration_data.left.image_points)> 1000 or key != -1:
            capturing = False

    calibration_data.image_size = l_gray.shape[::-1]
    calibration_data.left.image_size = l_gray.shape[::-1]
    calibration_data.right.image_size = r_gray.shape[::-1]

    cv2.destroyAllWindows()
    return calibration_data

def run_stereo_calibration(calibration_data:StereoCalibrationData):

    calibration = StereoCalibration(calibration_data)

    calibration.left.rmse, \
    calibration.left.camera_matrix, \
    calibration.left.distortion_coeffs, \
    calibration.left.rotations, \
    calibration.left.translations = cv2.calibrateCamera(
        objectPoints=calibration_data.left.object_points,
        imagePoints=calibration_data.left.image_points,
        imageSize=calibration.left.image_size,
        cameraMatrix=None,
        distCoeffs=None)

    calibration.right.rmse, \
    calibration.right.camera_matrix, \
    calibration.right.distortion_coeffs, \
    calibration.right.rotations, \
    calibration.right.translations = cv2.calibrateCamera(
        objectPoints=calibration_data.right.object_points,
        imagePoints=calibration_data.right.image_points,
        imageSize=calibration.right.image_size,
        cameraMatrix=None,
        distCoeffs=None)

    calibration.rmse, \
    calibration.left.refined_camera_matrix, \
    calibration.left.refined_distortion_coeffs, \
    calibration.right.refined_camera_matrix, \
    calibration.right.refined_distortion_coeffs, \
    calibration.rotation, \
    calibration.translation, \
    calibration.essential_matrix, \
    calibration.fundamental_matrix = cv2.stereoCalibrate(
        objectPoints=calibration_data.object_points,
        imagePoints1=calibration_data.left.image_points,
        imagePoints2=calibration_data.right.image_points,
        cameraMatrix1=calibration.left.camera_matrix,
        distCoeffs1=calibration.left.distortion_coeffs,
        cameraMatrix2=calibration.right.camera_matrix,
        distCoeffs2=calibration.right.distortion_coeffs,
        imageSize=calibration.image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=cv2.CALIB_FIX_INTRINSIC)

    calibration.left.rectification_transform, \
    calibration.right.rectification_transform, \
    calibration.left.projection_matrix, \
    calibration.right.projection_matrix, \
    calibration.reprojection_matrix, \
    calibration.left.roi, \
    calibration.right.roi = cv2.stereoRectify(
        cameraMatrix1=calibration.left.refined_camera_matrix,
        distCoeffs1=calibration.left.refined_distortion_coeffs,
        cameraMatrix2=calibration.right.refined_camera_matrix,
        distCoeffs2=calibration.right.refined_distortion_coeffs,
        imageSize=calibration.image_size,
        R=calibration.rotation,
        T=calibration.translation)

    calibration.left.rectification_maps  = cv2.initUndistortRectifyMap(
        cameraMatrix=calibration.left.refined_camera_matrix,
        distCoeffs=calibration.left.refined_distortion_coeffs,
        R=calibration.left.rectification_transform,
        newCameraMatrix=calibration.left.projection_matrix,
        size=calibration.left.image_size,
        m1type=cv2.CV_16SC2)
    calibration.right.rectification_maps = cv2.initUndistortRectifyMap(
        cameraMatrix=calibration.right.refined_camera_matrix,
        distCoeffs=calibration.right.refined_distortion_coeffs,
        R=calibration.right.rectification_transform,
        newCameraMatrix=calibration.right.projection_matrix,
        size=calibration.right.image_size,
        m1type=cv2.CV_16SC2)

    return calibration


def capture_data():
    calibration_data = capture_calibration_data()
    with open(EXPORT_DATA_FILENAME, "wb") as fp:
        pickle.dump(calibration_data, fp)
    
    print(calibration_data.to_string())

def load_data_and_calibrate():
    with open(IMPORT_DATA_FILENAME, "rb") as fp:
        calibration_data = pickle.load(fp)

    print("calibration data ready")
    print(calibration_data.to_string())

    calibration = run_stereo_calibration(calibration_data)
    with open(EXPORT_CALIBRATION_FILENAME, "wb") as fp:
        pickle.dump(calibration, fp)

    print("calibration completed")
    print(calibration.to_string())

def capture_data_and_calibrate():
    calibration_data = capture_calibration_data()
    with open(EXPORT_DATA_FILENAME, "wb") as fp:
        pickle.dump(calibration_data, fp)
    
    print("calibration data ready")
    print(calibration_data.to_string())

    calibration = run_stereo_calibration(calibration_data)
    with open(EXPORT_CALIBRATION_FILENAME, "wb") as fp:
        pickle.dump(calibration, fp)

    print("calibration completed")
    print(calibration.to_string())

def print_stored_data_and_calibration():
    with open(IMPORT_DATA_FILENAME, "rb") as fp:
        calibration_data = pickle.load(fp)

    print("calibration data ready")
    print(calibration_data.to_string())

    with open(IMPORT_CALIBRATION_FILENAME, "rb") as fp:
        calibration = pickle.load(fp)

    print("calibration completed")
    print(calibration.to_string())


if __name__ == "__main__":
    if "capture" in sys.argv:
        capture_data()
    if "calib" in sys.argv:
        load_data_and_calibrate()
    if "all" in sys.argv:
        capture_data_and_calibrate()
    if "show" in sys.argv:
        print_stored_data_and_calibration()