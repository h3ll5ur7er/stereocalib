import numpy as np
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
