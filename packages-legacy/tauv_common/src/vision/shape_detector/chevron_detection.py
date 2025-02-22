import numpy as np
from spatialmath import SE3, SO3
import cv2
from dataclasses import dataclass
from typing import Optional
from math import cos, sin

from vision.shape_detector.contour_filtering import filter_contours_by_bbox, approximate_contour, get_angles_contour
from tauv_util.cameras import CameraIntrinsics


@dataclass
class GetChevronPosesParams:
    min_size: (float, float)
    max_size: (float, float)
    min_aspect_ratio: float
    max_aspect_ratio: float
    contour_approximation_factor: float
    angles: (float, ...) # Pose relative to angles[0]
    angle_match_tolerance: float
    depth_window_size: int


def get_chevron_poses(mask: np.array, depth: np.array, intrinsics: CameraIntrinsics, params: GetChevronPosesParams, debug_img: Optional[np.array] = None) -> [SE3]:
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = filter_contours_by_bbox(contours, params.min_size, params.max_size, params.min_aspect_ratio, params.max_aspect_ratio)

    if debug_img is not None:
        cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 3)

    contours = [approximate_contour(contour, params.contour_approximation_factor) for contour in contours]

    contour_angles = [(contour, get_angles_contour(contour)) for contour in contours]

    points = []

    for contour, angles in contour_angles:
        if angles.shape[0] != len(params.angles):
            continue

        for start_i in range(angles.shape[0]):
            errors = np.abs(angles[np.arange(start_i, start_i + angles.shape[0]) % angles.shape[0], 0] - params.angles)

            if np.all(errors < params.angle_match_tolerance):
                points.append((tuple(contour[start_i, 0, :]), angles[start_i, 1]))

    if debug_img is not None:
        for point, angle in points:
            cv2.line(debug_img, point, (int(point[0] + 50 * cos(angle)), int(point[1] + 50 * sin(angle))), (0, 255, 0), 3)

    poses = []

    for point, angle in points:
        min_yx = (
            max(point[1] - params.depth_window_size // 2, 0),
            max(point[0] - params.depth_window_size // 2, 1),
        )
        max_yx = (
            min(point[1] + params.depth_window_size // 2, depth.shape[0]),
            min(point[0] + params.depth_window_size // 2, depth.shape[1]),
        )

        depth_window = depth[min_yx[0]:max_yx[0], min_yx[1]:max_yx[1]]
        depth_window = depth_window[depth_window > 0]
        if depth_window.shape[0] < 10:
            continue

        z = np.mean(depth_window)

        t = np.array([
            (point[0] - intrinsics.c_x) * (z / intrinsics.f_x),
            (point[1] - intrinsics.c_y) * (z / intrinsics.f_y),
            z,
        ])

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

    threshold_params = GetAdaptiveColorThresholdingParams(
        global_thresholds=np.array([
            [0, 0, 0, 20, 255, 255],
            [160, 0, 0, 180, 255, 255],
        ]),
        local_thresholds=np.array([
            [20, -255, 255, 255],
        ]),
        window_size=35
    )

    intrinsics = CameraIntrinsics(1035, 1035, 640, 360)

    plt.figure()
    plt.imshow(img[:, :, ::-1])

    plt.figure()
    plt.imshow(depth_img)
    plt.colorbar()

    mask = get_adaptive_color_thresholding(img, threshold_params)

    plt.figure()
    plt.imshow(mask)

    params = GetChevronPosesParams(
        min_size=(20, 20),
        max_size=(500, 500),
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.0,
        contour_approximation_factor=0.05,
        angles=(1.57, 2.35, 0.78, 1.57, 0.78, 2.35),
        angle_match_tolerance=0.1,
        depth_window_size=30
    )

    debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

    poses = get_chevron_poses(mask, depth_img, intrinsics, params, debug_img)

    plt.figure()
    plt.imshow(debug_img[:, :, ::-1])

    print(poses)

    plt.show()

if __name__ == "__main__":
    main()
