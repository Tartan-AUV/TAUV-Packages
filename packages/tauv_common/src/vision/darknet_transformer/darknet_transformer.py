#!/usr/bin/env python3

import rospy
import threading
from typing import Dict
from tauv_msgs.msg import FeatureDetection, FeatureDetections
import cv2
from tauv_msgs.srv import GetCameraInfo
import numpy as np
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image, CameraInfo
from vision.depth_estimation.depth_estimation import DepthEstimator
from tauv_util.transforms import tf2_transform_to_homogeneous
from geometry_msgs.msg import Vector3
import tf2_ros as tf2
from collections import deque
from spatialmath import SE3, SO3
from tauv_util.cameras import CameraIntrinsics
from vision.shape_detector.plane_fitting import fit_plane
from tauv_util.spatialmath import ros_transform_to_se3, r3_to_ros_vector3


class DarknetTransformer():

    def __init__(self):
        self._load_config()

        self._depth_lock: threading.Lock = threading.Lock()
        self._bboxes_lock: threading.Lock = threading.Lock()
        self._depth_lock.acquire()
        self._bboxes_lock.acquire()

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._cv_bridge: CvBridge = CvBridge()

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}
        self._depth_subs: Dict[str, rospy.Subscriber] = {}

        self._bboxes_queues: Dict[str, deque[BoundingBoxes]] = {}
        self._depth_queues: Dict[str, deque[Image]] = {}

        self._bboxes_sub: rospy.Subscriber = rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self._handle_bboxes)
        self._detections_pub: rospy.Publisher = rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)

        for frame_id in self._frame_ids:
            self._camera_infos[frame_id] = rospy.wait_for_message(f'vehicle/{frame_id}/depth/camera_info', CameraInfo, 60)
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(self._camera_infos[frame_id].K))

            self._depth_subs[frame_id] = rospy.Subscriber(f'vehicle/{frame_id}/depth/image_raw', Image, self._handle_depth, callback_args=frame_id)

            self._bboxes_queues[frame_id] = deque(maxlen=self._bboxes_queue_size)
            self._depth_queues[frame_id] = deque(maxlen=self._depth_queue_size)

        self._depth_lock.release()
        self._bboxes_lock.release()


    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _handle_depth(self, msg, frame_id):
        self._depth_lock.acquire()
        self._depth_queues[frame_id].append(msg)
        self._depth_lock.release()

    def _handle_bboxes(self, msg):
        self._bboxes_lock.acquire()
        frame_id = msg.image_header.frame_id
        self._bboxes_queues[frame_id].append(msg)
        self._bboxes_lock.release()

    def _update(self, timer_event):
        self._depth_lock.acquire()
        self._bboxes_lock.acquire()

        # TODO: needs something to remove bboxes after processing

        for frame_id in self._frame_ids:
            bboxes_queue = self._bboxes_queues[frame_id]
            depth_queue = self._depth_queues[frame_id]

            processed_bboxes_is = []

            for (bboxes_i, bboxes) in enumerate(bboxes_queue):
                deltas =\
                    [(depth_i, (msg.header.stamp - bboxes.image_header.stamp).to_sec())for (depth_i, msg) in enumerate(depth_queue)]
                deltas = list(filter(lambda t : abs(t[1]) < self._max_sync_delta, deltas))

                if len(deltas) == 0:
                    continue

                matched_depth_i, matched_depth_delta = min(deltas, key=lambda t : t[1])
                matched_depth = depth_queue[matched_depth_i]

                self._transform(frame_id, bboxes, matched_depth)

                processed_bboxes_is.append(bboxes_i)

            for processed_bboxes_i in reversed(sorted(processed_bboxes_is)):
                del bboxes_queue[processed_bboxes_i]

        self._depth_lock.release()
        self._bboxes_lock.release()

    def _transform(self, frame_id, bboxes, depth_msg):
        detections = FeatureDetections()
        detections.detector_tag = 'darknet'

        depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='mono16')
        depth = depth.astype(float) / 1000

        intrinsics = self._intrinsics[frame_id]

        for bbox in bboxes.bounding_boxes:
            # bbox_depth = DepthEstimator.estimate_absolute_depth(depth_img, bbox, self._camera_infos[frame_id])
            # bbox is xmin, xmax, ymin, ymax

            e_x = (bbox.xmin + bbox.xmax) // 2
            e_y = (bbox.ymin + bbox.ymax) // 2

            w = bbox.xmax - bbox.xmin
            h = bbox.ymax - bbox.ymin

            depth_mask = np.zeros(depth.shape, dtype=np.uint8)

            cv2.rectangle(
                depth_mask,
                (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
                (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
                255,
                -1
            )

            if np.sum(depth[(depth_mask > 0) & (depth > 0)]) < 10:
                continue

            z = np.mean(depth[(depth_mask > 0) & (depth > 0)])

            x = (e_x - intrinsics.c_x) * (z / intrinsics.f_x)
            y = (e_y - intrinsics.c_y) * (z / intrinsics.f_y)

            '''
            fit_plane_result = fit_plane(np.where(depth_mask, depth, 0), intrinsics)
            if fit_plane_result.n_points < 10:
                continue
            if fit_plane_result.normal[2] > 0:
                fit_plane_result.normal = -fit_plane_result.normal

            R = SO3.OA(np.array([0, -1, 0]), fit_plane_result.normal)

            a, b, c = fit_plane_result.coefficients
            z = c / (1 - a * (e_x - intrinsics.c_x) / intrinsics.f_x - b * (e_y - intrinsics.c_y) / intrinsics.f_y)
            x = (e_x - intrinsics.c_x) * (z / intrinsics.f_x)
            y = (e_y - intrinsics.c_y) * (z / intrinsics.f_y)
            '''

            t = np.array([x, y, z])

            R = SO3.OA(np.array([0, -1, 0]), np.array([0, 0, -1]))

            cam_t_detection = SE3.Rt(R, t)

            # TODO: Change this
            world_frame = f'{self._tf_namespace}/odom'
            camera_frame = f'{self._tf_namespace}/{frame_id}'

            try:
                transform = self._tf_buffer.lookup_transform(
                    world_frame,
                    camera_frame,
                    depth_msg.header.stamp,
                )

                odom_t_cam = ros_transform_to_se3(transform.transform)

                odom_t_detection = odom_t_cam * cam_t_detection

                detection = FeatureDetection()
                detection.tag = bbox.Class
                detection.position = r3_to_ros_vector3(odom_t_detection.t)
                detection.orientation = r3_to_ros_vector3(odom_t_detection.rpy())
                # Point towards camera
                detection.confidence = 1
                detection.SE2 = False
                detections.detections.append(detection)
            except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
                rospy.logwarn(f'Could not get transform from {world_frame} to {camera_frame}: {e}')

        self._detections_pub.publish(detections)

    def _load_config(self):
        self._frame_ids: [str] = rospy.get_param('~frame_ids')
        self._tf_namespace: str = rospy.get_param('tf_namespace')
        self._bboxes_queue_size = 10
        self._depth_queue_size = 10
        self._dt = 1 / 10
        self._max_sync_delta = 10.0
        self._max_tf_delta = 10.0

def main():
    rospy.init_node('darknet_transformer')
    n = DarknetTransformer()
    n.start()
