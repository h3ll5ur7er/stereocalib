#!/usr/bin/env python

import cv2
import sys
import numpy as np
import glob
import pickle
import uuid
from icecream import ic
# from settings import *
import settings
import preprocessing
from stereocam import StereoCamera

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)

class CameraCalibrationData:
    def __init__(self, uid):
        self.uid = uid
        self.image_size = None
        self.frames = []
        self.image_points = []
        self.object_points = []
    def to_string(self, indentation=0):
        base_indent = "\t"*indentation
        inner_indent = "\t"*(indentation+1)
        return "\n".join([
            f"{base_indent}CameraCalibrationData({self.uid}))",
            f"{inner_indent}image_size:  {self.image_size}",
            f"{inner_indent}frame_count: {len(self.frames)}",
            f"{inner_indent}ip_count:    {len(self.image_points)}",
            f"{inner_indent}op_count:    {len(self.object_points)}",
        ])

class StereoCalibrationData:
    def __init__(self, uid=None, left_uid=None, right_uid=None):
        self.uid = uid
        self.image_size = None
        self.left = CameraCalibrationData(left_uid)
        self.right = CameraCalibrationData(right_uid)
        self.object_points = []
    def to_string(self, indentation=0):
        base_indent = "\t"*indentation
        inner_indent = "\t"*(indentation+1)
        return "\n".join([
            f"{base_indent}StereoCalibration({self.uid}))",
            f"{inner_indent}image_size:  {self.image_size}",
            f"{inner_indent}op_count:    {len(self.object_points)}",
            f"{inner_indent}left:",
            self.left.to_string(indentation+2),
            f"{inner_indent}right:",
            self.right.to_string(indentation+2),
        ])

class CameraCalibration:
    def __init__(self, calibration_data:CameraCalibrationData):
        self.data_uid = calibration_data.uid
        self.image_size = calibration_data.image_size
        self.camera_matrix = None
        self.refined_camera_matrix = None
        self.distortion_coeffs = None
        self.refined_distortion_coeffs = None
        self.rotations = None
        self.translations = None
        self.rmse = None
        self.rectification_transform = None
        self.projection_matrix = None
        self.roi = None
        self.rectification_maps = None
    def to_string(self, indentation=0):
        base_indent = "\t"*indentation
        inner_indent = "\t"*(indentation+1)
        return "\n".join([
            f"{base_indent}CameraCalibration({self.data_uid}))",
            f"{inner_indent}image_size:  {self.image_size}",
            f"{inner_indent}rmse: {self.rmse}",
            f"{inner_indent}camera:",
            f"{inner_indent}\t{np.array2string(self.refined_camera_matrix[0,0])}\t{np.array2string(self.refined_camera_matrix[0,1])}\t{np.array2string(self.refined_camera_matrix[0,2])}",
            f"{inner_indent}\t{np.array2string(self.refined_camera_matrix[1,0])}\t{np.array2string(self.refined_camera_matrix[1,1])}\t{np.array2string(self.refined_camera_matrix[1,2])}",
            f"{inner_indent}\t{np.array2string(self.refined_camera_matrix[2,0])}\t{np.array2string(self.refined_camera_matrix[2,1])}\t{np.array2string(self.refined_camera_matrix[2,2])}",
            f"{inner_indent}distortion:",
            f"{inner_indent}\t{np.array2string(self.refined_distortion_coeffs[0,0])}\t{np.array2string(self.refined_distortion_coeffs[0,1])}\t{np.array2string(self.refined_distortion_coeffs[0,2])}\t{np.array2string(self.refined_distortion_coeffs[0,3])}\t{np.array2string(self.refined_distortion_coeffs[0,4])}",
        ])

class StereoCalibration:
    def __init__(self, calibration_data:StereoCalibrationData):
        self.data_uid = calibration_data.uid
        self.image_size = calibration_data.image_size
        self.left = CameraCalibration(calibration_data.left)
        self.right = CameraCalibration(calibration_data.right)
        self.rotation = None
        self.translation = None
        self.essential_matrix = None
        self.fundamental_matrix = None
        self.reprojection_matrix = None
        self.rmse = None
    def to_string(self, indentation=0):
        base_indent = "\t"*indentation
        inner_indent = "\t"*(indentation+1)
        return "\n".join([
            f"{base_indent}StereoCalibration({self.data_uid}))",
            f"{inner_indent}image_size:  {self.image_size}",
            f"{inner_indent}rmse: {self.rmse}",
            f"{inner_indent}left:",
            self.left.to_string(indentation+2),
            f"{inner_indent}right:",
            self.right.to_string(indentation+2),
        ])


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
    objp = np.zeros((1, settings.CHECKERBOARD[0] * settings.CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:settings.CHECKERBOARD[0], 0:settings.CHECKERBOARD[1]].T.reshape(-1, 2)
    objp*=scale
    return objp

def setup_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_RESOLUTION_HEIGHT)
    return cap

def draw_rect(img, x,y,w,h, color, mode="add"):
    new_img = img.copy()
    color = np.array(color, dtype=np.uint8)
    if mode == "add":
        new_img[y:y+h, x:x+w] += color
    elif mode == "set":
        new_img[y:y+h, x:x+w] = color
    return new_img

def capture_calibration_data(load_images=False, save_images=False):
    calibration_data = StereoCalibrationData(uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex)
    objp = generate_object_points(settings.CHECKERBOARD_SQUARE_SIZE_MM)
    cap = setup_camera(settings.CAMERA_ID)
    l_viz = None
    r_viz = None
    capturing = True
    if load_images:
        iterable = glob.glob('./images/l_*.png')
    else:
        iterable = StereoCamera().frame_generator()

    for img in iterable:
        if isinstance(img, str):
            rname = img.replace("l_", "r_")
            l_img = cv2.imread(img)
            r_img = cv2.imread(rname)
        else:
            if settings.FLIP_CAMERA:
                img = img[::-1, ::-1, :]
            l_img, r_img = img[:, :settings.FRAME_WIDTH, :], img[:, settings.FRAME_WIDTH:, :]

        if l_viz is None:
            l_viz = np.zeros_like(l_img)
            cv2.imshow('l_viz',l_viz)
        if r_viz is None:
            r_viz = np.zeros_like(r_img)
            cv2.imshow('r_viz',r_viz)

        l_gray = cv2.cvtColor(src=l_img, code=cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(src=r_img, code=cv2.COLOR_BGR2GRAY)

        l_found, l_corners = cv2.findChessboardCorners(
            l_gray,
            settings.CHECKERBOARD,
            flags=settings.CHECKERBOARD_FLAGS)
        r_found, r_corners = cv2.findChessboardCorners(
            r_gray,
            settings.CHECKERBOARD,
            flags=settings.CHECKERBOARD_FLAGS)

        if l_found and r_found:
            l_corners2 = cv2.cornerSubPix(l_gray, l_corners, (11,11),(-1,-1), settings.SUB_PIXEL_CRITERIA)
            r_corners2 = cv2.cornerSubPix(r_gray, r_corners, (11,11),(-1,-1), settings.SUB_PIXEL_CRITERIA)
            l_roi = get_checkerboard_aabb(l_corners2)
            l_blur = calculate_blurryness(l_img, *l_roi, lbl="l")
            r_roi = get_checkerboard_aabb(r_corners2)
            r_blur = calculate_blurryness(r_img, *r_roi, lbl="r")

            if l_blur < settings.BLURRYNESS_THRESHOLD or r_blur < settings.BLURRYNESS_THRESHOLD:
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
            l_corners_img = cv2.drawChessboardCorners(l_gray, settings.CHECKERBOARD, l_corners2, l_found)
            r_corners_img = cv2.drawChessboardCorners(r_gray, settings.CHECKERBOARD, r_corners2, r_found)

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
            break

    ic()
    calibration_data.image_size = l_gray.shape[::-1]
    calibration_data.left.image_size = l_gray.shape[::-1]
    calibration_data.right.image_size = r_gray.shape[::-1]
    ic()
    cv2.destroyAllWindows()
    ic()
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
        criteria=settings.STEREO_CALIB_CRITERIA,
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
    ic()
    with open(settings.EXPORT_DATA_FILENAME, "wb") as fp:
        pickle.dump(calibration_data, fp)
    
    print(calibration_data.to_string())

def load_data_and_calibrate():
    with open(settings.IMPORT_DATA_FILENAME, "rb") as fp:
        calibration_data = pickle.load(fp)

    print("calibration data ready")
    print(calibration_data.to_string())

    calibration = run_stereo_calibration(calibration_data)
    with open(settings.EXPORT_CALIBRATION_FILENAME, "wb") as fp:
        pickle.dump(calibration, fp)

    print("calibration completed")
    print(calibration.to_string())

def capture_data_and_calibrate():
    calibration_data = capture_calibration_data()
    with open(settings.EXPORT_DATA_FILENAME, "wb") as fp:
        pickle.dump(calibration_data, fp)
    
    print("calibration data ready")
    print(calibration_data.to_string())

    calibration = run_stereo_calibration(calibration_data)
    with open(settings.EXPORT_CALIBRATION_FILENAME, "wb") as fp:
        pickle.dump(calibration, fp)

    print("calibration completed")
    print(calibration.to_string())

def print_stored_data_and_calibration():
    with open(settings.IMPORT_DATA_FILENAME, "rb") as fp:
        calibration_data = pickle.load(fp)

    print("calibration data ready")
    print(calibration_data.to_string())

    with open(settings.IMPORT_CALIBRATION_FILENAME, "rb") as fp:
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
