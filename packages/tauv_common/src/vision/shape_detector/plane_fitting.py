import numpy as np
from dataclasses import dataclass
from tauv_util.cameras import CameraIntrinsics
from spatialmath.base import R3
from math import exp


@dataclass
class FitPlaneResult:
    normal: R3
    coefficients: R3
    error: float
    n_points: int


def fit_plane(depth: np.array, intrinsics: CameraIntrinsics) -> FitPlaneResult:
    img_points = depth.nonzero()

    if np.sum(depth > 0) < 10:
        return FitPlaneResult(np.zeros(3), np.zeros(3), 0, 0)

    img_points = np.column_stack(img_points)

    z = depth[img_points[:, 0], img_points[:, 1]]

    points = np.column_stack((
        (z / intrinsics.f_x) * (img_points[:, 1] - intrinsics.c_x),
        (z / intrinsics.f_y) * (img_points[:, 0] - intrinsics.c_y),
        z,
    ))

    n_points = points.shape[0]

    A = np.column_stack((points[:, 0], points[:, 1], np.ones(n_points)))
    b = points[:, 2]

    coeffs, r, _, _ = np.linalg.lstsq(A, b, rcond=None)

    normal = np.array([coeffs[0], coeffs[1], -1])
    normal = normal / np.linalg.norm(normal)

    result = FitPlaneResult(normal, coeffs, r, n_points)

    return result
