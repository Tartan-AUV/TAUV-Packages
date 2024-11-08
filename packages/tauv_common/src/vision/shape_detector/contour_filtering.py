import numpy as np
import cv2
from math import pi, sin, cos, atan2


def filter_contours_by_bbox(contours: np.array, min_size: (float, float), max_size: (float, float), min_aspect_ratio: float, max_aspect_ratio: float) -> np.array:
    filtered_contours = []

    for contour in contours:
        (_, _,), (w, h), _ = cv2.minAreaRect(contour)
        size = (min(w, h), max(w, h))
        if min_size[0] <= size[0] <= max_size[0] and min_size[1] <= size[1] <= max_size[1]\
            and min_aspect_ratio <= size[1] / size[0] <= max_aspect_ratio:
            filtered_contours.append(contour)

    return filtered_contours


def filter_contours_by_defects(contours: [np.array], templates: [np.array], defect_threshold: float) -> [np.array]:
    filtered_contours = []

    for contour, template in zip(contours, templates):
        defects = np.zeros(template.shape[0])

        for i in range(template.shape[0]):
            point = (int(template[i, 0]), int(template[i, 1]))
            defects[i] = abs(cv2.pointPolygonTest(contour, point, measureDist=True))

        _, _, w, h = cv2.boundingRect(template.astype(int))
        size = (w + h) / 2

        mean_defect = np.mean(defects) / size

        if mean_defect < defect_threshold:
            filtered_contours.append(contour)

    return filtered_contours


def fit_ellipse_contour(contour: np.array, n_points: int) -> np.array:
    ellipse = cv2.fitEllipse(contour)
    (e_y, e_x), (e_w, e_h), e_theta_deg = ellipse
    e_theta = -np.deg2rad(e_theta_deg)

    theta = np.linspace(0, 2 * pi, n_points)

    ellipse_contour = np.zeros((n_points, 2))
    ellipse_contour[:, 0] =\
        e_y + (e_h / 2) * np.cos(theta) * sin(e_theta) + (e_w / 2) * np.sin(theta) * cos(e_theta)
    ellipse_contour[:, 1] =\
        e_x + (e_h / 2) * np.cos(theta) * cos(e_theta) - (e_w / 2) * np.sin(theta) * sin(e_theta)

    return ellipse_contour.astype(int)


def approximate_contour(contour: np.array, factor: float) -> np.array:
    length = cv2.arcLength(contour, True)
    contour_approx =  cv2.approxPolyDP(contour, length * factor, True)
    return contour_approx


def get_angles_contour(contour: np.array) -> np.array:
    n_points = contour.shape[0]

    angles = np.zeros((n_points, 2))

    for i in range(n_points):
        x1 = contour[(i - 1) % n_points, 0, :]
        x2 = contour[i, 0, :]
        x3 = contour[(i + 1) % n_points, 0, :]

        v1 = x1 - x2
        v2 = x3 - x2

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot_product / norm_product)

        bisector = v1 + v2
        bisector_angle = (atan2(bisector[1], bisector[0]) + pi) % (2 * pi)

        angles[i, 0] = angle
        angles[i, 1] = bisector_angle

    return angles
