#!/usr/bin/env python3

# Author: Gleb Ryabtsev, 2023

import rospy
import numpy as np
import tf2_ros
import cv2
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Quaternion
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from tauv_util.parms import Parms
from tauv_util.transforms import quat_to_rpy, tf2_transform_to_homogeneous, tf2_transform_to_quat, multiply_quat
from tauv_util.types import tm
from math import atan2

from vision.detectors.gate_detector import GateDetector

class GateDetectorNode:
    def __init__(self):
        parameters = Parms(rospy.get_param("~gate_detector_parameters"))
        print(parameters)
        self._detector = GateDetector(parameters)

        self._rgb_sub = message_filters.Subscriber('color', Image, queue_size=10)
        self._depth_sub = message_filters.Subscriber('depth', Image, queue_size=10)
        self._camera_info_sub = message_filters.Subscriber('camera_info',
                                                           CameraInfo, queue_size=10)

        self._camera_info = rospy.wait_for_message('camera_info', CameraInfo, 60)

        self._time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub,
                self._depth_sub,
             ], queue_size=10, slop=0.5)
        self._time_synchronizer.registerCallback(self.callback)

        self._detection_pub = rospy.Publisher('global_map/feature_detections', FeatureDetections,
                                              queue_size=10)
        self._tf_namespace = rospy.get_param('tf_namespace')

        self._bridge = cv_bridge.CvBridge()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

    def callback(self, rgb: Image, depth: Image):
        # Update camera matrix
        self._detector.set_camera_matrix(self._camera_info.K)

        # Convert colors
        cv_rgb = self._bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_depth = self._bridge.imgmsg_to_cv2(depth,
                                              desired_encoding="passthrough")
        cv_depth = cv_depth.astype(np.float32) / 1000 # Integer mm -> float m

        tf_cam_odom = self._tf_buffer.lookup_transform(
            f'{self._tf_namespace}/odom',
            f'{self._tf_namespace}/oakd_front',
        rospy.Time(0)
        )

        H_cam_odom = tf2_transform_to_homogeneous(tf_cam_odom)
        q_cam_odom = tf2_transform_to_quat(tf_cam_odom)

        # Get detections
        detections = self._detector.detect(cv_rgb, cv_depth)
        print(detections)

        detections_msg = FeatureDetections()
        # detections_msg.header.stamp = rgb.header.stamp
        detections_msg.detector_tag = 'gate_detector'

        # Need to transform these detections into world frame :(

        for pos in detections:
            position_odom = H_cam_odom @ np.array([pos.x, pos.y, pos.z, 1])
            position_odom = position_odom[0:3] / position_odom[3]

            orientation_odom = multiply_quat(q_cam_odom, np.array([pos.q0, pos.q1, pos.q2, pos.q3]))

            yaw = atan2(position_odom[1] - tf_cam_odom.transform.translation.y, position_odom[0] - tf_cam_odom.transform.translation.x)

            orientation_odom_rpy = quat_to_rpy(tm(orientation_odom, Quaternion))
            detection_msg = FeatureDetection()
            detection_msg.tag = 'gate'
            detection_msg.position = Point(position_odom[0], position_odom[1], position_odom[2])
            detection_msg.orientation = Point(0, 0, yaw)
            # detection_msg.count = 1
            detections_msg.detections.append(detection_msg)

        print(detections_msg)
        self._detection_pub.publish(detections_msg)


def main():
    rospy.init_node('gate_detector', anonymous=True)
    g = GateDetectorNode()
    rospy.spin()

main()
