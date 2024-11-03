import numpy as np
from spatialmath import SE3, SO3
import cv2
from dataclasses import dataclass
from typing import Optional
from math import pi, atan2

from vision.shape_detector.contour_filtering import filter_contours_by_bbox
from tauv_util.cameras import CameraIntrinsics


@dataclass
class GetPathMarkerPosesParams:
    min_size: (float, float)
    max_size: (float, float)
    min_aspect_ratio: float
    max_aspect_ratio: float


def get_path_marker_poses(mask: np.array, depth: np.array, intrinsics: CameraIntrinsics,
                          params: GetPathMarkerPosesParams, debug_img: Optional[np.array] = None) -> [SE3]:
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if debug_img is not None:
        cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 3)

    contours = filter_contours_by_bbox(contours, params.min_size, params.max_size, params.min_aspect_ratio, params.max_aspect_ratio)

    if debug_img is not None:
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)

    poses = []

    for contour in contours:
        bbox = cv2.minAreaRect(contour)

        (b_x, b_y), (b_w, b_h), b_theta_deg = bbox
        b_theta = np.deg2rad(b_theta_deg)

        depth_mask = np.zeros(mask.shape, dtype=np.uint8)

        cv2.drawContours(depth_mask, [contour], -1, color=255, thickness=-1)

        depth_window = depth[(depth_mask > 0) & (depth > 0)]
        if depth_window.shape[0] < 10:
            continue
        z = np.mean(depth_window)
        x = (b_x - intrinsics.c_x) * (z / intrinsics.f_x)
        y = (b_y - intrinsics.c_y) * (z / intrinsics.f_y)

        t = np.array([x, y, z])

        angle = b_theta
        if b_h > b_w:
            angle = b_theta + pi / 2

        center_angle = atan2(b_y - intrinsics.c_y, b_x - intrinsics.c_x)
        if abs(angle - center_angle) > pi / 2:
            angle = (angle + pi)

        angle = angle % (2 * pi)

        pose = SE3.Rt(SO3.Rz(angle), t)

        rvec, _ = cv2.Rodrigues(pose.R)
        tvec = pose.t

        if debug_img is not None:
            cv2.drawFrameAxes(debug_img, intrinsics.to_matrix(), np.zeros(5), rvec, tvec, 0.1)

        poses.append(pose)

    return poses
