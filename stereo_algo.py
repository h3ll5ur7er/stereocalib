import cv2
import numpy as np
import pickle
import image_output as io
from importlib import reload
import plyfile
from icecream import ic
import settings
import calibration_tool
from status_service import StatusService


def calculate_disparity(l_img, r_img, grab):
    reload(settings)
    with open(settings.ACTIVE_CALIBRATION_FILENAME, "rb") as fp:
        calibration = pickle.load(fp)
    reload(io)
    reload(plyfile)

    l_img_remap = cv2.remap(l_img, calibration.left.rectification_maps[0],  calibration.left.rectification_maps[1],  cv2.INTER_LANCZOS4)
    r_img_remap = cv2.remap(r_img, calibration.right.rectification_maps[0], calibration.right.rectification_maps[1], cv2.INTER_LANCZOS4)

    if settings.MATCHER == "bm":
        left_matcher  = cv2.StereoBM_create(**settings.BM_SETTINGS)
    elif settings.MATCHER == "sgbm":
        left_matcher  = cv2.StereoSGBM_create(**settings.SGBM_SETTINGS)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)

    left_disp = left_matcher.compute(l_img_remap, r_img_remap)
    right_disp = right_matcher.compute(r_img_remap, l_img_remap)

    if settings.MAX_DISPARITY:
        left_disp = np.clip(left_disp, 0, settings.MAX_DISPARITY)
        right_disp = np.clip(right_disp, -settings.MAX_DISPARITY, 0)

    wls_filter.setLambda(settings.WLS_LAMBDA)
    wls_filter.setSigmaColor(settings.WLS_SIGMA_COLOR)
    filtered_disp = wls_filter.filter(
        disparity_map_left=left_disp,
        left_view=l_img_remap,
        disparity_map_right=right_disp,
        right_view=r_img_remap).astype(np.float32)

    if settings.DEBUG:
        io.show(l_img, "l")
        io.show(r_img, "r")
        io.show(l_img_remap, "lm")
        io.show(r_img_remap, "rm")
        io.show(io.normalize(left_disp), "dl")
        io.show(io.normalize(right_disp), "dr")
        io.show(io.normalize(io.edges(left_disp)), "ld")
        io.show(io.normalize(io.edges(right_disp)), "rd")
        io.show(filtered_disp, "d")
        io.show(io.normalize(filtered_disp), "dn")
        io.show(io.normalize(io.edges(filtered_disp)), "de")

    if grab:
        StatusService().status = "exporting pcl"
        depth_map = cv2.reprojectImageTo3D(filtered_disp, calibration.reprojection_matrix)
        ic(depth_map.shape)
        plyfile.export(depth_map, filtered_disp, l_img_remap)

    io.show(
        io.group(
            io.concat(io.put_name(filtered_disp, "disp F"),             io.put_name(io.normalize(io.edges(filtered_disp)), "disp F e"), io.put_name(io.normalize(filtered_disp), "disp F n"),         io.put_name(np.zeros_like(l_img), "overlay")),
            io.concat(io.put_name(io.normalize(left_disp), "disp L n"), io.put_name(io.normalize(io.edges(left_disp)), "disp L e"),     io.put_name(io.normalize(io.edges(right_disp)), "disp R e"),  io.put_name(io.normalize(right_disp), "disp R n")),
            io.concat(io.put_name(l_img, "raw L"),                      io.put_name(l_img_remap, "calib L"),                            io.put_name(r_img_remap, "calib R"),                          io.put_name(r_img, "raw R")),
        ), "asd")

    return filtered_disp
