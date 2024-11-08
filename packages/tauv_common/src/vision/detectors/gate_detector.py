#!/usr/bin/env python3

# Author: Gleb Ryabtsev, 2023

import cv2
import numpy as np
from numpy import NaN
from skspatial.objects import Plane, Line
from scipy.spatial.transform import Rotation
from vision.shape_detector.adaptive_color_thresholding import get_adaptive_color_thresholding, GetAdaptiveColorThresholdingParams

import math
import itertools

from collections import namedtuple
from dataclasses import dataclass
from tauv_util.parms import Parms

from typing import List

class GateDetector:
    GatePosition = namedtuple('GatePosition', 'x y z q0 q1 q2 q3')

    @dataclass
    class _GateCandidate:
        left_pole_img: np.ndarray
        right_pole_img: np.ndarray
        img_translation: np.ndarray
        img_rotation: Rotation
        depth_translation: np.ndarray = None
        depth_rotation: Rotation = None
        depth_plane_dev: float = None

    def __init__(self, parms: Parms):
        self._parms = parms
        self._camera_matrix = np.array([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, 1]])
        self._camera_matrix_inv = self._inv_camera_matrix(self._camera_matrix)
        # self._act_params = GetAdaptiveColorThresholdingParams(
        #     self._parms.preprocessing.global_thresholds,
        #     self._parms.preprocessing.local_thresholds,
        #     self._parms.preprocessing.window_size
        # )


    def set_camera_matrix(self, camera_matrix):
        """
        Set camera parameters
        @param camera_matrix: Intrinsics matrix (3x3)
        @return: None
        """
        self._camera_matrix = np.array(camera_matrix).reshape(3, 3)
        # get inverse
        self._camera_matrix_inv = self._inv_camera_matrix(self._camera_matrix)

    def detect(self, color: np.array, depth: np.array):

        p = GetAdaptiveColorThresholdingParams(
            global_thresholds=[[0,0,0,255,255,255]],
            local_thresholds=[[-255,-255,255,-5]],
            window_size=35
        )

        print(f"{color.shape=}, {depth.shape=}")

        depth = cv2.resize(depth, color.shape[:2][::-1])

        print(f"{color.shape=}, {depth.shape=}")

        filtered = get_adaptive_color_thresholding(color, p)*255

        cv2.imshow('depth', depth)
        cv2.imshow('filtered', filtered)

        cv2.waitKey(1)

        # return []
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        dilated = cv2.dilate(filtered, kernel)
        eroded = cv2.erode(dilated, kernel)
        edges = cv2.Canny(eroded, 100, 200)

        # blurred = cv2.blur(edges, (5, 5))

        # cv2.imshow('blurred', blurred)

        lines = cv2.HoughLinesP(eroded, 1, math.pi/180, 150, minLineLength=100, maxLineGap=5)

        if lines is None:
            return []

        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2-y1, x2-x1)
            HIGH = math.pi/2 + 0.1
            LOW = math.pi/2 - 0.1

            if LOW <= abs(angle) <= HIGH:
                vertical_lines.append(line)
                cv2.line(color, (x1, y1), (x2, y2), (0, 255, 0), 2)



        vertical_lines = np.array(vertical_lines)

        if len(vertical_lines) < 2:
            return []

        # sort vertical lines by x1
        vertical_lines = vertical_lines[vertical_lines[:, 0, 0].argsort()]

        MID_ALLOWED_DEV = 0.1
        MIN_DIST = 100

        vertical_candidate_pairs = []
        for i in range(len(vertical_lines)):
            for j in range(len(vertical_lines)):
                if i == j:
                    continue

                x1 = vertical_lines[i, 0, 0]
                x2 = vertical_lines[j, 0, 0]
                mid = (x1 + x2) / 2
                d = abs(x2-x1)
                if d < MIN_DIST:
                    continue

                diff = np.abs(vertical_lines[i+1:j, 0, 0] - mid)
                # print(diff)
                if np.any(diff < MID_ALLOWED_DEV*d):
                    # draw center circle
                    x_mid = int((x1 + x2) / 2)
                    y_mid = int((vertical_lines[i, 0, 1] + vertical_lines[j, 0, 1]) / 2)
                    cv2.circle(color, (x_mid, y_mid), 10, (0, 0, 255), 2)
                    vertical_candidate_pairs.append((i, j))

        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2-y1, x2-x1)
            HIGH = 1.0
            LOW = -1.0

            if LOW <= angle <= HIGH:
                horizontal_lines.append(line)
                cv2.line(color, (x1, y1), (x2, y2), (0, 255, 0), 2)


        horizontal_lines = np.array(horizontal_lines)

        #sort horizontal lines by y1 in reverse order
        horizontal_lines = horizontal_lines[horizontal_lines[:, 0, 1].argsort()[::-1]]

        # differentiate y1
        y1_diff = np.diff(horizontal_lines[:, 0, 1])

        # find first index where y1_diff is less than Y_CLUSTER_MAX_DIST
        Y_CLUSTER_MAX_DIST = 20
        y1_diff_idx = np.where(y1_diff < Y_CLUSTER_MAX_DIST)[0][0]
        print(y1_diff_idx)

        top_line = horizontal_lines[y1_diff_idx]

        cv2.line(color, (top_line[0, 0], top_line[0, 1]), (top_line[0, 2], top_line[0, 3]), (0, 0, 255), 2)

        gate_candidates = []

        for i1, i2 in vertical_candidate_pairs:
            # find intersections with top line
            intersection1 = GateDetector.find_intersection(vertical_lines[i1, 0], top_line[0])
            intersection2 = GateDetector.find_intersection(vertical_lines[i2, 0], top_line[0])
            if intersection1 is None or intersection2 is None:
                continue

            #convert to int
            intersection1 = (int(intersection1[0]), int(intersection1[1]))
            intersection2 = (int(intersection2[0]), int(intersection2[1]))

            depth_array = self._get_line_array(np.array(intersection1), np.array(intersection2),
                                               depth)[:,2]
            x1,y1 = intersection1
            x2,y2 = intersection2

            # check if intersection is in the image
            if x1 < 0 or x1 >= depth.shape[1] or y1 < 0 or y1 >= depth.shape[0] or \
                x2 < 0 or x2 >= depth.shape[1] or y2 < 0 or y2 >= depth.shape[0]:
                continue


            d1 = depth[y1, x1]
            d2 = depth[y2, x2]


            if d1 == 0 or d2 == 0 or d1 == NaN or d2 == NaN:
                continue

            mat = self._camera_matrix_inv
            # 4 element np array
            pt1 = np.array([0,0,0,0])
            pt2 = np.array([0,0,0,0])
            pt1[0] = x1*d1
            pt1[1] = y1*d1
            pt1[2] = d1
            pt1[3] = 1.0
            pt2[0] = x2*d2
            pt2[1] = y2*d2
            pt2[2] = d2
            pt2[3] = 1.0
            # pt2 = np.array([x2, y2, d2, 1])
            pt13 = np.matmul(mat, pt1.transpose())
            pt23 = np.matmul(mat, pt2.transpose())

            #get distance between pt13 and 23
            dist = np.linalg.norm(pt13 - pt23)

            gate_candidates.append((pt13, pt23, dist))

            cv2.circle(color, intersection1, 10, (0, 0, 255), 2)
            cv2.circle(color, intersection2, 10, (0, 0, 255), 2)



        bestCandidate = None
        bestDev = 0
        GATE_WIDTH = 1.8
        for (pt1, pt2, dist) in gate_candidates:
            if bestCandidate is None or (bestDev > abs(dist - GATE_WIDTH)):
                bestCandidate = (pt1, pt2, dist)
                bestDev = abs(dist - GATE_WIDTH)

        print(bestCandidate, bestDev)

        # get midpoint
        if bestCandidate == None:
            return []
        midpoint = (bestCandidate[0] + bestCandidate[1]) / 2


        cv2.imshow('rgb', color)

        cv2.waitKey(1)

        # for c in candidates:
        #     d = self.GatePosition(*c.depth_translation, *c.depth_rotation.as_quat())
        #     detections.append(d)
        return [self.GatePosition(midpoint[0], midpoint[1], midpoint[2], 0, 0, 0, 1)]

    @staticmethod
    def find_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Check if lines are parallel
        if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0:
            return None  # Lines are parallel, no intersection point

        # Calculate intersection point coordinates using Cramer's rule
        det_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        det_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        intersection_x = det_x / denominator
        intersection_y = det_y / denominator

        return intersection_x, intersection_y

    def _find_candidates(self, img):
        """
        Find gate candidates in a color image.
        @param img: OpenCV BGR image
        @return: List of 3D positions of gate candidates.
        """
        p = self._parms

        # Filter in HSV, fill holes and detect contours
        # filtered = self._get_color_filter_mask(img,
        #                                        p.preprocessing.hsv_boundaries)
        print(self._act_params)
        filtered = get_adaptive_color_thresholding(img, self._act_params)

        cv2.imshow("img", img)

        dilated = self._dilate(filtered)
        eroded = self._erode(dilated)

        blurred = cv2.GaussianBlur(eroded*255, (3, 3), 0.5)
        cv2.imshow("blurred", blurred)
        lines = cv2.HoughLinesP(blurred, 1, 2*math.pi/180, 300, minLineLength=100, maxLineGap=5)

        self._draw_lines(img, lines, 2, (255, 0, 0))

        cv2.imshow("lines", cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        cv2.waitKey(1)
        # return []
        if len(lines) == 0:
            return []

        candidates = []
        for i1, i2 in itertools.combinations(range(len(lines)), 2):
            l1 = lines[i1, 0]
            l2 = lines[i2, 0]

            if l1[0] < l2[0] and l1[2] < l2[2]:
                left, right = l1, l2
            elif l2[0] < l1[0] and l2[2] < l1[2]:
                left, right = l2, l1
            else:
                # intersection
                continue

            retval, tvec, rot = self._pose_from_side_poles(left, right)
            if retval:
                candidates.append(
                    GateDetector._GateCandidate(left, right, tvec, rot))

        return candidates


    def _filter_by_depth_dev(self,
                             candidates: List[_GateCandidate]):
        """
        Filters a list of _GateCandidates based on the deviation from the
        best fit plane
        @param candidates: List of GateCandidates
        @return: List of GateCandidates
        """
        p = self._parms

        result = []
        for c in candidates:
            if c.depth_plane_dev is None:
                # Could not fit plane, so assuming candidate is invalid
                continue
            if c.depth_plane_dev < p.depth.dev_cutoff:
                result.append(c)

        return result

    def _calculate_depth_poses(self,
                               candidates: List[_GateCandidate],
                               depth_map: np.ndarray):
        """
        Calculate the depth-based translations and rotations of _GateCandidate
        instances
        @param candidates: Array of gate candidates
        @param depth_map: Depth image (1 channel)
        @return: None
        """
        p = self._parms.depth

        for gc in candidates:

            # Sample lines from the depth map
            left_top, left_bottom = gc.left_pole_img[:2], gc.left_pole_img[2:]
            right_top, right_bottom = gc.right_pole_img[:2], gc.right_pole_img[2:]
            left_2d = self._get_line_array(left_top, left_bottom, depth_map)
            right_2d = self._get_line_array(right_top, right_bottom, depth_map)
            pts_2d = np.concatenate((left_2d, right_2d), axis=0)
            pts_2d = pts_2d[::p.sampling_interval]

            # Remove points with depth=0
            depth = pts_2d[:, 2]
            index = np.where(depth == 0.0)
            pts_2d = np.delete(pts_2d, index, axis=0)

            # Remove outliers
            depth = pts_2d[:, 2]
            if len(depth) < 3:
                continue
            quantiles = np.quantile(depth, [.25, .75])
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower = q1 - p.outlier_f * iqr
            upper = q3 + p.outlier_f * iqr
            index = np.where(np.logical_or(depth < lower, upper < depth))
            pts_2d = np.delete(pts_2d, index, axis=0)

            # Prepare array of 2D points for matrix multiplication
            # pts_2d_4: [[u*depth, v*depth, depth, 1], ...]
            pts_2d_4 = np.zeros((len(pts_2d), 4), dtype=np.float64)
            pts_2d_4[:, 0] = pts_2d[:, 0] * pts_2d[:, 2]
            pts_2d_4[:, 1] = pts_2d[:, 1] * pts_2d[:, 2]
            pts_2d_4[:, 2] = pts_2d[:, 2]
            pts_2d_4[:, 3] = 1

            # get 3D points (camera_matrix_inv x pts_2d_4)
            mat = self._camera_matrix_inv
            pts_3d = np.matmul(mat, pts_2d_4.transpose())
            pts_3d = pts_3d.transpose()

            # fit a plane
            try:
                plane = Plane.best_fit(pts_3d)
            except ValueError:
                continue

            # calculate deviation
            a, b, c, d = plane.cartesian()
            normal = np.array([a, b, c])
            normal_length = np.linalg.norm(normal)
            normal = normal / normal_length
            d /= normal_length
            plane_pt = normal * d * -1.0
            squared_distances = np.abs((pts_3d - plane_pt).dot(normal))
            gc.depth_plane_dev = np.sum(squared_distances) / len(squared_distances)

            # calculate gate position based on the plane
            # get the gate center point vector
            center_2d = (left_top + left_bottom + right_top + right_bottom) / 4.0
            center_2d_4 = np.concatenate((center_2d, [1, 1]))
            center_3d_vec = np.matmul(mat, center_2d_4)[:3]
            center_3d_line = Line([0., 0., 0.], center_3d_vec)
            gc.depth_translation = np.array(plane.intersect_line(center_3d_line))

            # calculate rotation
            if normal[2] < 0.0:
                normal *= (-1.0)
            z_unit = [0, 0, 1]
            x, y, z = np.cross(z_unit, normal)
            w = 1.0 + np.dot(normal, z_unit)
            gc.depth_rotation = Rotation.from_quat((x, y, z, w))

    def _filter_by_pose_diff(self, candidates):
        """
        Filters candidates list by the difference between poses calcualted from
        the depth and the color images
        @param candidates: List of _GateCandidate instances
        @return: List of _GateCandidate instances
        """
        p = self._parms

        # Prepare thresholds (faster to compare squares)
        max_translation_diff_sq = p.depth.max_translation_diff ** 2
        max_rotation_diff_sq = math.radians(p.depth.max_rotation_diff) ** 2

        # Squared norm function
        sq_euclidean = lambda x: np.inner(x, x)

        result = []
        for c in candidates:
            if c.depth_translation is None or c.depth_rotation is None:
                continue
            translation_diff_sq = sq_euclidean(c.img_translation - c.depth_translation)
            rotation_diff = c.depth_rotation.inv() * c.img_rotation
            angle_sq = sq_euclidean(rotation_diff.as_rotvec())
            if translation_diff_sq > max_translation_diff_sq:
                continue
            if angle_sq > max_rotation_diff_sq:
                continue
            result.append(c)

        return result

    def _clip_candidates(self, candidates):
        """
        Eliminates candidates with positions and orientations outside clipping
        range. Considers only GateCandidate.img_pos and GateCandidate.img_rot
        @param candidates: List of _GateCandidate instances
        @return: List of _GateCandidate instances
        """
        p = self._parms

        result = []
        for c in candidates:
            # clip x (depth) and orientation

            # angle of rotation around rotation vector
            angle = np.linalg.norm(c.img_rotation.as_rotvec())
            if ((c.img_translation[2] < p.clipping.max_distance) and
                    (math.degrees(angle) < p.clipping.max_angle)):
                result.append(c)

        return result

    def _lines_from_contours(self, contours):
        """
        For an array of CV contours, calculate centerlines of their smallest
        bounding rectangles.
        @param contours: Array of contours
        @return: Array of lines, shape=(n,4), dtype=int32, first point is
        guaranteed to be above the second
        """
        p = self._parms

        lines = []
        for i, c in enumerate(contours):
            (cx, cy), (w, h), th = cv2.minAreaRect(c)

            # Fix the orientation
            if w > h:
                th -= 90.0
            w, h = min(w, h), max(w, h)

            # filter based on minimum size
            if (w == 0 or
                    h < p.contours.min_size or
                    h / w < p.contours.min_hw_ratio):
                continue

            th = -math.radians(th)
            dx = math.sin(th) * h / 2
            dy = math.cos(th) * h / 2

            lines.append([[cx - dx, cy - dy, cx + dx, cy + dy]])

        lines = np.array(lines).astype(int)
        return lines

    def _dilate(self, img):
        ks = self._parms.preprocessing.dilation_ks
        print(f'{ks=}')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*ks + 1, 2*ks + 1), (ks, ks))
        dilated = cv2.dilate(img, kernel)
        return dilated

    def _erode(self, img):
        """
        Erodes the image.
        @param img: Binary image
        @return: Binary image, eroded
        """
        es = self._parms.preprocessing.erosion_ks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (2 * es + 1, 2 * es + 1), (es, es))
        eroded = cv2.erode(img, kernel)
        return eroded

    def _fill_holes(self, img):
        """
        Fills holes in a binary mask.
        @param img (binary mask, 1 channel)
        @return: binary mask with filled holes
        """
        ds = self._parms.preprocessing.dilation_ks

        # dilate the image to close small holes
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (2 * ds + 1, 2 * ds + 1),
                                                    (ds, ds))
        dilated = cv2.dilate(img, dilation_kernel)

        # flood fill the image
        im_floodfill = dilated.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        floodfilled = dilated | im_floodfill_inv

        return floodfilled

    def _pose_from_side_poles(self, l1, l2):
        """
        Compute the gate pose from two lines in the image corresponding to the
        side poles.
        @param l1: Line 1: [x1, y1, x2, y2]
        @param l2: Line 2, should be to the right of line 1
        @return: tuple of retval, translation vec, and rotation
        (as scipy.spatial.transform.Rotation)
        """
        p = self._parms
        image_points = np.array(
            [l1[:2], l2[:2], l2[2:], l1[2:]]
        ).astype(np.float64)
        object_points = np.array(p.geometry.side_poles)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                          self._camera_matrix,
                                          None,
                                          flags=cv2.SOLVEPNP_IPPE)
        # x, y axes same the those of the image (x - right, y - down)
        # z points forward
        if not retval:
            return False, None, None
        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        rot = Rotation.from_rotvec(rvec)
        return True, tvec, rot

    @staticmethod
    def _inv_camera_matrix(mat):
        """
        Calculate the inverse of a camera matrix
        @param mat: 3x3 np camera matrix
        @return: 3x4 inverted camera matrix
        """
        inverse_3x3 = np.linalg.inv(mat)
        inverse_3x4 = np.concatenate((inverse_3x3, [[0], [0], [0]]), axis=1)
        return inverse_3x4

    @staticmethod
    def _get_color_filter_mask(img, hsv_boundaries):
        """
        Set all pixels with hsv values outside the given range to zero.
        @param: img: Image in BGR
        @param: hsv_boundaries [[[hmin, smin, vmin], [hmax, smax, vmax]], ...]
        @return: Binary mask of image areas within color ranges
        """
        hsv_boundaries_arr = np.array(hsv_boundaries, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        for boundary in hsv_boundaries_arr:
            new_mask = cv2.inRange(hsv, boundary[0], boundary[1])
            mask = cv2.bitwise_or(mask, new_mask)
        return mask

    @staticmethod
    def _get_line_array(P1, P2, img):
        # https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
        # define local variables for readability

        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(
                    int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[
            (colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[
            itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

        return itbuffer

    @staticmethod
    def _draw_lines(img, lines, width, color):
        if lines is None:
            return
        for line in lines:
            line = line[0]
            if len(line) == 4:
                cv2.line(img, tuple(line[0:2]), tuple(line[2:4]), color, width)
            else:
                raise NotImplementedError()