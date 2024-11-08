import numpy as np
from spatialmath import SE3, SO3
import cv2
from dataclasses import dataclass
from typing import Optional
from math import pi, atan2

from vision.shape_detector.contour_filtering import filter_contours_by_bbox, approximate_contour
from tauv_util.cameras import CameraIntrinsics


@dataclass
class GetLidPosesParams:
    orange_min_size: (float, float)
    orange_max_size: (float, float)
    orange_min_aspect_ratio: float
    orange_max_aspect_ratio: float
    orange_contour_approximation_factor: float
    purple_min_size: (float, float)
    purple_max_size: (float, float)
    purple_min_aspect_ratio: float
    purple_max_aspect_ratio: float


def get_lid_poses(orange_mask: np.array, purple_mask: np.array, depth: np.array, intrinsics: CameraIntrinsics,
                  params: GetLidPosesParams, debug_img: Optional[np.array] = None) -> [SE3]:
    kernel_open = np.ones((3, 3), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel_open)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel_open)

    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    orange_contours = filter_contours_by_bbox(orange_contours, params.orange_min_size, params.orange_max_size, params.orange_min_aspect_ratio, params.orange_max_aspect_ratio)

    orange_contours = [approximate_contour(contour, params.orange_contour_approximation_factor) for contour in orange_contours]

    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    purple_contours = filter_contours_by_bbox(purple_contours, params.purple_min_size, params.purple_max_size, params.purple_min_aspect_ratio, params.purple_max_aspect_ratio)

    if debug_img is not None:
        cv2.drawContours(debug_img, orange_contours, -1, (255, 0, 0), 3)
        cv2.drawContours(debug_img, purple_contours, -1, (255, 0, 0), 3)

    poses = []

    for orange_contour in orange_contours:
        if orange_contour.shape[0] != 4:
            continue

        purple_outside = []
        for purple_contour in purple_contours:
            (x, y), (_, _), _ = cv2.minAreaRect(purple_contour)

            outside = cv2.pointPolygonTest(orange_contour, (x, y), False) == -1
            purple_outside.append(outside)

        if all(purple_outside):
            continue

        depth_mask = np.zeros(depth.shape, dtype=np.uint8)

        cv2.drawContours(depth_mask, [orange_contour], -1, 255, -1)

        (b_x, b_y), (b_w, b_h), b_theta_deg = cv2.minAreaRect(orange_contour)
        b_theta = np.deg2rad(b_theta_deg)

        depth = depth[(depth > 0) & (depth_mask > 0)]
        if depth.shape[0] < 10:
            continue

        z = np.mean(depth)

        t = np.array([
            (b_x - intrinsics.c_x) * (z / intrinsics.f_x),
            (b_y - intrinsics.c_y) * (z / intrinsics.f_y),
            z,
        ])

        angle = b_theta
        if b_w > b_h:
            angle = b_theta + pi / 2

        center_angle = atan2(b_y - intrinsics.c_y, b_x - intrinsics.c_x)
        if abs(angle - center_angle) > pi / 2:
            angle = (angle + pi)

        angle = angle % (2 * pi)

        R = SO3.Rz(angle)

        pose = SE3.Rt(R, t)

        poses.append(pose)

        if debug_img is not None:
            rvec, _ = cv2.Rodrigues(pose.R)
            tvec = pose.t
            cv2.drawFrameAxes(debug_img, intrinsics.to_matrix(), np.zeros(5), rvec, tvec, 0.1)

    return poses


def main():
    import sys
    from vision.shape_detector.adaptive_color_thresholding import GetAdaptiveColorThresholdingParams, get_adaptive_color_thresholding
    import matplotlib.pyplot as plt

    img_path = sys.argv[1]
    depth_path = sys.argv[2]

    img = cv2.imread(img_path)
    depth_img = cv2.imread(depth_path, -1)
    depth_img = depth_img.astype(np.float32) / 1000

    orange_threshold_params = GetAdaptiveColorThresholdingParams(
        global_thresholds=np.array([
            [0, 50, 50, 20, 255, 255],
        ]),
        local_thresholds=np.array([
            [20, 0, 255, 255],
        ]),
        window_size=70
    )
    purple_threshold_params = GetAdaptiveColorThresholdingParams(
        global_thresholds=np.array([
            [120, 50, 50, 130, 255, 255],
        ]),
        local_thresholds=np.array([
            [-255, -255, 255, 255],
        ]),
        window_size=35
    )

    intrinsics = CameraIntrinsics(1035, 1035, 640, 360)

    plt.figure()
    plt.imshow(img[:, :, ::-1])

    plt.figure()
    plt.imshow(depth_img)
    plt.colorbar()

    orange_mask = get_adaptive_color_thresholding(img, orange_threshold_params)
    purple_mask = get_adaptive_color_thresholding(img, purple_threshold_params)

    plt.figure()
    plt.imshow(orange_mask)

    plt.figure()
    plt.imshow(purple_mask)

    params = GetLidPosesParams(
        orange_min_size=(100, 100),
        orange_max_size=(500, 500),
        orange_min_aspect_ratio=1.5,
        orange_max_aspect_ratio=2.5,
        orange_contour_approximation_factor=0.05,
        purple_min_size=(10, 50),
        purple_max_size=(500, 500),
        purple_min_aspect_ratio=8.0,
        purple_max_aspect_ratio=10.0,
    )

    debug_img = cv2.cvtColor(255 * orange_mask, cv2.COLOR_GRAY2BGR)

    poses = get_lid_poses(orange_mask, purple_mask, depth_img, intrinsics, params, debug_img)

    plt.figure()
    plt.imshow(debug_img[:, :, ::-1])

    print(poses)

    plt.show()



if __name__ == "__main__":
    main()
