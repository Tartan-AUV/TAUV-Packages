import numpy as np
import cv2
from math import atan2, pi
from enum import IntEnum


def threshold(img, hsv_ranges):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w, _ = img_hsv.shape

    img_thresh = np.zeros((h, w), dtype=np.uint8)
    for hsv_range in hsv_ranges:
        low = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
        high = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
        mask = cv2.inRange(img_hsv, low, high)
        img_thresh = img_thresh | mask

    return img_thresh


def get_border_mask(img, left, right, top, bottom):
    mask = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    if left > 0:
        mask[:, :left] = 0
    if right > 0:
        mask[:, -right:] = 0
    if top > 0:
        mask[:top, :] = 0
    if bottom > 0:
        mask[-bottom:, :] = 0

    return mask


def clean(img, close_size, open_size):
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_open = np.ones((open_size, open_size), np.uint8)

    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open)

    return img_opened


def get_components(img, area_threshold):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    label = 0
    out_labels = np.zeros(labels.shape, dtype=np.uint8)

    for i in range(n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            out_labels[labels == i] = label

            label += 1

    return out_labels, label


def get_contour(img, approximation_factor):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]

    perimeter = cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, approximation_factor * perimeter, True)
    hull = cv2.convexHull(contour_approx)

    contour_approx_flat = np.fliplr(contour_approx.reshape((contour_approx.shape[0], contour_approx.shape[2])))
    hull_flat = np.fliplr(hull.reshape((hull.shape[0], hull.shape[2])))

    return contour_approx_flat, hull_flat


def get_angles(contour):
    n_points = contour.shape[0]

    angles = np.zeros((n_points))

    for i in range(n_points):
        y1, x1 = contour[(i - 1) % n_points, :]
        y2, x2 = contour[i, :]
        y3, x3 = contour[(i + 1) % n_points, :]

        angle = atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2)

        angle = (angle + pi) % (2 * pi)

        if angle > pi:
            angle = angle - pi

        angles[i] = angle

    return angles


class AngleClassification(IntEnum):
    NONE = 0
    FRONT = 1
    BACK = 2
    SIDE = 3
    TAIL = 4


def classify_angles(contour, hull, angles, angle_threshold, mask):
    n_points = contour.shape[0]

    classifications = np.zeros((n_points), dtype=np.uint8)

    for i in range(n_points):
        classification = AngleClassification.NONE

        point = contour[i]
        angle = angles[i]

        hull_list = hull.tolist()

        if mask[point[0], point[1]] != 0:
            if np.abs(angle - (pi / 2)) < angle_threshold:
                if point.tolist() in hull_list:
                    classification = AngleClassification.FRONT
                else:
                    classification = AngleClassification.BACK
            elif np.abs(angle - (3 * pi / 4)) < angle_threshold and point.tolist() in hull_list:
                classification = AngleClassification.SIDE
            elif np.abs(angle - (pi / 4)) < angle_threshold and point.tolist() in hull_list:
                classification = AngleClassification.TAIL

        classifications[i] = classification.value

    return classifications
