import numpy as np
from spatialmath import SE3, SO3
import cv2
from dataclasses import dataclass
from typing import Optional

from vision.shape_detector.contour_filtering import filter_contours_by_bbox, filter_contours_by_defects, fit_ellipse_contour
from vision.shape_detector.plane_fitting import fit_plane
from tauv_util.cameras import CameraIntrinsics


@dataclass
class GetCirclePosesParams:
    min_size: (float, float)
    max_size: (float, float)
    min_aspect_ratio: float
    max_aspect_ratio: float
    depth_mask_scale: float


def get_circle_poses(mask: np.array, depth: np.array, intrinsics: CameraIntrinsics, params: GetCirclePosesParams, debug_img: Optional[np.array] = None) -> [SE3]:
    kernel_open = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_open)
    kernel_close = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if debug_img is not None:
        cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 1)

    contours = filter_contours_by_bbox(contours, params.min_size, params.max_size, params.min_aspect_ratio, params.max_aspect_ratio)

    templates = [fit_ellipse_contour(contour, 32) for contour in contours]

    # TODO: This is slow

    contours = filter_contours_by_defects(contours, templates, 0.1)

    poses = []

    for contour in contours:
        ellipse = cv2.fitEllipse(contour)
        (e_x, e_y), (e_w, e_h), _ = ellipse

        depth_mask = np.zeros(mask.shape, dtype=np.uint8)

        size = (e_w + e_h) / 2

        cv2.rectangle(
            depth_mask,
            (int(e_x - (size * params.depth_mask_scale) / 2), int(e_y - (size * params.depth_mask_scale) / 2)),
            (int(e_x + (size * params.depth_mask_scale) / 2), int(e_y + (size * params.depth_mask_scale) / 2)),
            255,
            -1
        )

        fit_plane_result = fit_plane(np.where(depth_mask, depth, 0), intrinsics)
        if fit_plane_result.n_points < 10:
            continue
        if fit_plane_result.normal[2] > 0:
            fit_plane_result.normal = -fit_plane_result.normal

        R = SO3.OA(np.array([0, -1, 0]), fit_plane_result.normal)

        # TODO: wrap this in a function that gives pose on plane that projects to an image point
        a, b, c = fit_plane_result.coefficients
        z = c / (1 - a * (e_x - intrinsics.c_x) / intrinsics.f_x - b * (e_y - intrinsics.c_y) / intrinsics.f_y)
        x = (e_x - intrinsics.c_x) * (z / intrinsics.f_x)
        y = (e_y - intrinsics.c_y) * (z / intrinsics.f_y)

        t = np.array([x, y, z])

        pose = SE3.Rt(R, t)

        rvec, _ = cv2.Rodrigues(pose.R)
        tvec = pose.t

        if debug_img is not None:
            cv2.drawFrameAxes(debug_img, intrinsics.to_matrix(), np.zeros(5), rvec, tvec, 0.1)

        poses.append(pose)

    return poses


def main():
    import sys
    from adaptive_color_thresholding import get_adaptive_color_thresholding
    import matplotlib.pyplot as plt

    img_path = sys.argv[1]
    depth_path = sys.argv[2]

    img = cv2.imread(img_path)
    depth_img = cv2.imread(depth_path, -1)
    depth_img = depth_img.astype(np.float32) / 1000

    global_thresholds = np.array([
        [0, 0, 50, 20, 255, 150],
        [160, 0, 0, 180, 255, 255],
    ])

    local_thresholds = np.array([
        [20, -255, 255, 255],
    ])

    window_size = 35

    intrinsics = CameraIntrinsics(1035, 1035, 640, 360)

    plt.figure()
    plt.imshow(img[:, :, ::-1])

    plt.figure()
    plt.imshow(depth_img)
    plt.colorbar()

    mask = get_adaptive_color_thresholding(img, global_thresholds, local_thresholds, window_size)

    plt.figure()
    plt.imshow(mask)

    params = GetCirclePosesParams(
        min_size=(50, 50),
        max_size=(500, 500),
        min_aspect_ratio=1.0,
        max_aspect_ratio=1.2,
        depth_mask_scale=2.0,
    )

    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    poses = get_circle_poses(mask, depth_img, intrinsics, params, debug_img)

    plt.figure()
    plt.imshow(debug_img[:, :, ::-1])

    print(poses)

    plt.show()


if __name__ == "__main__":
    main()
