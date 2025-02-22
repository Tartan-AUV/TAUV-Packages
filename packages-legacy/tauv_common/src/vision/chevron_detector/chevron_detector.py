import rospy
import threading
import collections
import cv2
import numpy as np
from typing import Optional
import tf2_ros as tf2
import cv_bridge
from sensor_msgs.msg import CameraInfo, Image
from tauv_msgs.msg import FeatureDetections, FeatureDetection
from tauv_util.transforms import tf2_transform_to_homogeneous, tf2_transform_to_quat, multiply_quat, quat_to_rpy
from geometry_msgs.msg import Point, Quaternion
from math import cos, sin
from vision.chevron_detector.chevron_detector_processing import *
from scipy.spatial.transform import Rotation


class ChevronDetector:

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lock.acquire()

        self._load_config()

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._cv_bridge: cv_bridge.CvBridge = cv_bridge.CvBridge()

        self._color_camera_info: CameraInfo = rospy.wait_for_message(f'vehicle/{self._frame_id}/color/camera_info', CameraInfo, 60)
        self._depth_camera_info: CameraInfo = rospy.wait_for_message(f'vehicle/{self._frame_id}/depth/camera_info', CameraInfo, 60)

        self._img_sub: rospy.Subscriber = rospy.Subscriber(f'vehicle/{self._frame_id}/color/image_raw', Image, self._handle_img)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber(f'vehicle/{self._frame_id}/depth/image_raw', Image, self._handle_depth)
        self._detections_pub: rospy.Publisher = rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)
        self._debug_pub: rospy.Publisher = rospy.Publisher('vision/chevron_detector/debug', Image, queue_size=10)

        self._depth: Optional[Image] = None

        self._lock.release()

    def start(self):
        rospy.spin()

    def _handle_depth(self, msg):
        self._lock.acquire()
        self._depth = msg
        self._lock.release()

    def _handle_img(self, msg):
        self._lock.acquire()

        if self._depth is None:
            self._lock.release()
            return

        img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        depth_img = self._cv_bridge.imgmsg_to_cv2(self._depth, desired_encoding='mono16')

        detections = FeatureDetections()
        detections.detector_tag = 'chevron_detector'

        tf_cam_odom_H = None
        tf_cam_odom_rotm = None
        world_frame = f'{self._tf_namespace}/odom'
        camera_frame = f'{self._tf_namespace}/{self._frame_id}'
        try:
            tf_cam_odom = self._tf_buffer.lookup_transform(
                world_frame,
                camera_frame,
                msg.header.stamp,
                rospy.Duration(0.1)
            )
            tf_cam_odom_H = tf2_transform_to_homogeneous(tf_cam_odom)
            tf_cam_odom_rotm = Rotation.from_quat(tf2_transform_to_quat(tf_cam_odom)).as_matrix()
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn(f'Could not get transform from {world_frame} to {camera_frame}: {e}')
            self._lock.release()
            return

        img_threshold = threshold(img, self._hsv_ranges)

        img_clean = clean(img_threshold, self._close_size, self._open_size)

        suction_mask = get_border_mask(img_clean, 0, self._suction_mask_width, 0, 0)

        img_clean = img_clean & suction_mask

        debug_img = cv2.cvtColor(img_clean, cv2.COLOR_GRAY2BGR)

        labels, n_labels = get_components(img_clean, self._area_threshold)

        valid_mask = get_border_mask(
            img_clean,
            self._border_mask_width,
            self._suction_mask_width + self._border_mask_width,
            self._border_mask_width,
            self._border_mask_width
        )

        for i in range(1, n_labels):
            component = np.where(labels == i, 255, 0).astype(np.uint8)

            contour, hull = get_contour(component, self._contour_approximation_factor)

            n_points = contour.shape[0]

            angles = get_angles(contour)
            classifications = classify_angles(contour, hull, angles, self._angle_threshold, valid_mask)

            for j in range(n_points):
                if classifications[j] == AngleClassification.FRONT:
                    cv2.circle(debug_img, (contour[j, 1], contour[j, 0]), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(debug_img, (contour[j, 1], contour[j, 0]), 5, (255, 0, 0), -1)

            if np.sum(classifications == AngleClassification.FRONT) == 1:
                front_i = classifications.tolist().index(AngleClassification.FRONT)

                y1, x1 = contour[(front_i - 1) % n_points].astype(np.int)
                y2, x2 = contour[front_i].astype(np.int)
                y3, x3 = contour[(front_i + 1) % n_points].astype(np.int)

                angle1 = atan2(y2 - y3, x2 - x3)
                angle2 = atan2(y2 - y1, x2 - x1)
                angle = (angle1 + angle2) / 2

                if min(angle1, angle2) < (-pi / 2) and max(angle1, angle2) > (pi / 2):
                    angle = angle + pi

                cv2.line(debug_img,
                         (x2, y2),
                         (int(x2 + 100 * cos(angle)), int(y2 + 100 * sin(angle))),
                         (0, 0, 255),
                         5
                         )

                if self._estimate_table_depth:
                    depth_window = depth_img[y2 - (self._depth_window_size // 2):y2 + (self._depth_window_size // 2),
                                   x2 - (self._depth_window_size // 2):x2 + (self._depth_window_size // 2)
                                   ]
                    depth = (np.nanmean(depth_window) / 1000) * 1.5
                else:
                    depth = self._table_depth - tf_cam_odom_H[2, 3]

                if not np.isnan(depth):
                    fx = self._color_camera_info.K[0]
                    cx = self._color_camera_info.K[2]
                    fy = self._color_camera_info.K[4]
                    cy = self._color_camera_info.K[5]

                    x = ((x2 - cx) * depth) / (1.5 * fx)
                    y = ((y2 - cy) * depth) / (1.5 * fy)

                    world_point_h = tf_cam_odom_H @ np.array([x, y, depth, 1])

                    world_point = world_point_h[0:3] / world_point_h[3]

                    cam_rotm = Rotation.from_euler('ZYX', np.array([angle, 0, 0])).as_matrix()
                    world_rotm = tf_cam_odom_rotm @ cam_rotm
                    world_yaw = Rotation.from_matrix(world_rotm).as_euler('ZYX')[0]

                    detection = FeatureDetection()
                    detection.header.stamp = rospy.Time()
                    detection.tag = 'chevron'
                    detection.position = Point(world_point[0], world_point[1], world_point[2])
                    detection.orientation = Point(0, 0, world_yaw)
                    detection.confidence = 1.0
                    detection.SE2 = False
                    detections.detections.append(detection)
            else:
                print('bad detection')

        self._detections_pub.publish(detections)

        debug_img_msg = self._cv_bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')

        self._debug_pub.publish(debug_img_msg)

        self._lock.release()

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_id = rospy.get_param('~frame_id')
        self._hsv_ranges = rospy.get_param('~hsv_ranges')
        self._close_size = rospy.get_param('~close_size')
        self._open_size = rospy.get_param('~open_size')
        self._area_threshold = rospy.get_param('~area_threshold')
        self._suction_mask_width = rospy.get_param('~suction_mask_width')
        self._border_mask_width = rospy.get_param('~border_mask_width')
        self._contour_approximation_factor = rospy.get_param('~contour_approximation_factor')
        self._angle_threshold = rospy.get_param('~angle_threshold')
        self._depth_window_size = rospy.get_param('~depth_window_size')
        self._table_depth = rospy.get_param('~table_depth')
        self._estimate_table_depth = rospy.get_param('~estimate_table_depth')


def main():
    rospy.init_node('chevron_detector')
    n = ChevronDetector()
    n.start()