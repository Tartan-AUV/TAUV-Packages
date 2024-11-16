import rospy
import threading
from functools import partial
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tauv_msgs.msg import FeatureDetections, FeatureDetection
import cv2
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from tauv_util.spatialmath import r3_to_ros_point
from tauv_util.cameras import CameraIntrinsics, CameraDistortion
import message_filters
from transform_client import TransformClient
from spatialmath import SE3
from vision.shape_detector.circle_detection import GetCirclePosesParams, get_circle_poses
from vision.shape_detector.path_marker_detection import GetPathMarkerPosesParams, get_path_marker_poses
from vision.shape_detector.chevron_detection import GetChevronPosesParams, get_chevron_poses
from vision.shape_detector.lid_detection import GetLidPosesParams, get_lid_poses
from vision.shape_detector.adaptive_color_thresholding import GetAdaptiveColorThresholdingParams, get_adaptive_color_thresholding


class ShapeDetector:

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lock.acquire()

        self._load_config()

        self._tf_client: TransformClient = TransformClient()

        self._cv_bridge: CvBridge = CvBridge()

        self._camera_infos: {str: CameraInfo} = {}
        self._intrinsics: {str: CameraIntrinsics} = {}
        self._distortions: {str: CameraDistortion} = {}
        for frame_id in self._frame_ids:
            rospy.loginfo(f'Waiting for {frame_id} camera info')
            info = rospy.wait_for_message(f'vehicle/{frame_id}/color/camera_info', CameraInfo, 60)
            self._camera_infos[frame_id] = info
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(info.K))
            # self._distortions[frame_id] = CameraDistortion.from_matrix(np.array(info.D))
            self._distortions[frame_id] = CameraDistortion.from_matrix(np.zeros(5))

        rospy.loginfo(f'Got camera infos')

        self._synchronizers: {str: message_filters.ApproximateTimeSynchronizer} = {}
        self._circle_debug_img_pubs: {str: rospy.Publisher} = {}
        self._path_marker_debug_img_pubs: {str: rospy.Publisher} = {}
        self._chevron_debug_img_pubs: {str: rospy.Publisher} = {}
        self._lid_debug_img_pubs: {str: rospy.Publisher} = {}
        for frame_id in self._frame_ids:
            color_sub = message_filters.Subscriber(f'vehicle/{frame_id}/color/image_raw', Image)
            depth_sub = message_filters.Subscriber(f'vehicle/{frame_id}/depth/image_raw', Image)

            synchronizer = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=self._synchronizer_queue_size, slop=self._synchronizer_slop
            )
            synchronizer.registerCallback(partial(self._handle_imgs, frame_id=frame_id))
            self._synchronizers[frame_id] = synchronizer

            if 'circle' in self._detectors[frame_id]:
                self._circle_debug_img_pubs[frame_id] =\
                    rospy.Publisher(f'vision/shape_detector/{frame_id}/circle/debug_image', Image, queue_size=10)
            if 'path_marker' in self._detectors[frame_id]:
                self._path_marker_debug_img_pubs[frame_id] = \
                    rospy.Publisher(f'vision/shape_detector/{frame_id}/path_marker/debug_image', Image, queue_size=10)
            if 'chevron' in self._detectors[frame_id]:
                self._chevron_debug_img_pubs[frame_id] = \
                    rospy.Publisher(f'vision/shape_detector/{frame_id}/chevron/debug_image', Image, queue_size=10)
            if 'lid' in self._detectors[frame_id]:
                self._lid_debug_img_pubs[frame_id] = \
                    rospy.Publisher(f'vision/shape_detector/{frame_id}/lid/debug_image', Image, queue_size=10)

        self._detections_pub: rospy.Publisher =\
            rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)

        self._reload_config_server: rospy.Service = rospy.Service(f'vision/shape_detector/reload_config', Trigger, self._handle_reload_config)

        self._lock.release()

    def start(self):
        rospy.spin()

    def _handle_imgs(self, color_msg: Image, depth_msg: Image, frame_id: str):
        with self._lock:
            color = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='mono16')
            depth = depth.astype(float) / 1000

            world_frame = f'{self._tf_namespace}/odom'
            camera_frame = f'{self._tf_namespace}/{frame_id}'

            world_t_cam = self._tf_client.get_a_to_b(world_frame, camera_frame, color_msg.header.stamp)

            detections = FeatureDetections()
            detections.detector_tag = 'shape_detector'

            if 'circle' in self._detectors[frame_id]:
                circle_detections, circle_debug_img = self._get_circle_detections(color, depth, world_t_cam,
                                                                self._intrinsics[frame_id])
                detections.detections = detections.detections + circle_detections
                circle_debug_msg = self._cv_bridge.cv2_to_imgmsg(circle_debug_img, encoding='bgr8')
                self._circle_debug_img_pubs[frame_id].publish(circle_debug_msg)

            if 'path_marker' in self._detectors[frame_id]:
                path_marker_detections, path_marker_debug_img = self._get_path_marker_detections(color, depth, world_t_cam, self._intrinsics[frame_id])
                detections.detections = detections.detections + path_marker_detections
                path_marker_debug_msg = self._cv_bridge.cv2_to_imgmsg(path_marker_debug_img, encoding='bgr8')
                self._path_marker_debug_img_pubs[frame_id].publish(path_marker_debug_msg)

            if 'chevron' in self._detectors[frame_id]:
                chevron_detections, chevron_debug_img = self._get_chevron_detections(color, depth, world_t_cam, self._intrinsics[frame_id])
                detections.detections = detections.detections + chevron_detections
                chevron_debug_msg = self._cv_bridge.cv2_to_imgmsg(chevron_debug_img, encoding='bgr8')
                self._chevron_debug_img_pubs[frame_id].publish(chevron_debug_msg)

            if 'lid' in self._detectors[frame_id]:
                lid_detections, lid_debug_img = self._get_lid_detections(color, depth, world_t_cam, self._intrinsics[frame_id])
                detections.detections = detections.detections + lid_detections
                lid_debug_msg = self._cv_bridge.cv2_to_imgmsg(lid_debug_img, encoding='bgr8')
                self._lid_debug_img_pubs[frame_id].publish(lid_debug_msg)

            self._detections_pub.publish(detections)

    def _handle_reload_config(self, req: TriggerRequest) -> TriggerResponse:
        res = TriggerResponse()

        with self._lock:
            self._load_config()

        res.success = True
        res.message = 'success'

        return res

    def _get_circle_detections(self, color: np.array, depth: np.array, world_t_cam: SE3,
                               intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        mask = get_adaptive_color_thresholding(color, self._circle_threshold_params)

        debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

        cam_t_circles = get_circle_poses(mask, depth, intrinsics, self._circle_pose_params, debug_img)

        world_t_circles = [world_t_cam * cam_t_circle for cam_t_circle in cam_t_circles]

        detections = []
        for world_t_circle in world_t_circles:
            detection = FeatureDetection()

            detection.confidence = 1
            detection.tag = 'circle'
            detection.SE2 = False
            detection.position = r3_to_ros_point(world_t_circle.t)
            # TODO: enforce verticality
            detection.orientation = r3_to_ros_point(world_t_circle.rpy())

            detections.append(detection)

        return detections, debug_img

    def _get_path_marker_detections(self, color: np.array, depth: np.array, world_t_cam: SE3,
                                    intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        mask = get_adaptive_color_thresholding(color, self._path_marker_threshold_params)

        debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

        cam_t_path_markers = get_path_marker_poses(mask, depth, intrinsics, self._path_marker_pose_params, debug_img)

        world_t_path_markers = [world_t_cam * cam_t_path_marker for cam_t_path_marker in cam_t_path_markers]

        detections = []
        for world_t_path_marker in world_t_path_markers:
            detection = FeatureDetection()

            detection.confidence = 1
            detection.tag = 'path_marker'
            detection.SE2 = False
            detection.position = r3_to_ros_point(world_t_path_marker.t)
            rpy = world_t_path_marker.rpy()
            rpy[0:2] = 0
            detection.orientation = r3_to_ros_point(rpy)

            detections.append(detection)

        return detections, debug_img

    def _get_chevron_detections(self, color: np.array, depth: np.array, world_t_cam: SE3, intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        mask = get_adaptive_color_thresholding(color, self._chevron_threshold_params)

        debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

        cam_t_chevrons = get_chevron_poses(mask, depth, intrinsics, self._chevron_pose_params, debug_img)

        world_t_chevrons = [world_t_cam * cam_t_chevron for cam_t_chevron in cam_t_chevrons]

        detections = []

        for world_t_chevron in world_t_chevrons:
            detection = FeatureDetection()

            detection.confidence = 1
            detection.tag = 'chevron'
            detection.SE2 = False
            detection.position = r3_to_ros_point(world_t_chevron.t)
            rpy = world_t_chevron.rpy()
            rpy[0:2] = 0
            detection.orientation = r3_to_ros_point(rpy)

            detections.append(detection)

        return detections, debug_img

    def _get_lid_detections(self, color: np.array, depth: np.array, world_t_cam: SE3, intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        orange_mask = get_adaptive_color_thresholding(color, self._lid_orange_threshold_params)
        purple_mask = get_adaptive_color_thresholding(color, self._lid_purple_threshold_params)

        debug_img = cv2.cvtColor(255 * (orange_mask | purple_mask), cv2.COLOR_GRAY2BGR)

        cam_t_lids = get_lid_poses(orange_mask, purple_mask, depth, intrinsics, self._lid_pose_params, debug_img)

        world_t_lids = [world_t_cam * cam_t_lid for cam_t_lid in cam_t_lids]

        detections = []

        for world_t_lid in world_t_lids:
            detection = FeatureDetection()

            detection.confidence = 1
            detection.tag = 'lid'
            detection.SE2 = False
            detection.position = r3_to_ros_point(world_t_lid.t)
            rpy = world_t_lid.rpy()
            rpy[0:2] = 0
            detection.orientation = r3_to_ros_point(rpy)

            detections.append(detection)

        return detections, debug_img

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_ids = rospy.get_param('~frame_ids')
        self._detectors: {str: [str]} = {}
        for frame_id in self._frame_ids:
            self._detectors[frame_id] = rospy.get_param(f'~detectors/{frame_id}')
        self._synchronizer_queue_size = rospy.get_param('~synchronizer_queue_size')
        self._synchronizer_slop = rospy.get_param('~synchronizer_slop')

        self._circle_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~circle/threshold/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~circle/threshold/local_thresholds')),
            window_size=rospy.get_param('~circle/threshold/window_size'),
        )
        self._circle_pose_params: GetCirclePosesParams = GetCirclePosesParams(
            min_size=tuple(rospy.get_param('~circle/pose/min_size')),
            max_size=tuple(rospy.get_param('~circle/pose/max_size')),
            min_aspect_ratio=rospy.get_param('~circle/pose/min_aspect_ratio'),
            max_aspect_ratio=rospy.get_param('~circle/pose/max_aspect_ratio'),
            depth_mask_scale=rospy.get_param('~circle/pose/depth_mask_scale'),
        )

        self._path_marker_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~path_marker/threshold/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~path_marker/threshold/local_thresholds')),
            window_size=rospy.get_param('~path_marker/threshold/window_size'),
        )
        self._path_marker_pose_params: GetPathMarkerPosesParams = GetPathMarkerPosesParams(
            min_size=tuple(rospy.get_param('~path_marker/pose/min_size')),
            max_size=tuple(rospy.get_param('~path_marker/pose/max_size')),
            min_aspect_ratio=rospy.get_param('~path_marker/pose/min_aspect_ratio'),
            max_aspect_ratio=rospy.get_param('~path_marker/pose/max_aspect_ratio'),
        )

        self._chevron_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~chevron/threshold/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~chevron/threshold/local_thresholds')),
            window_size=rospy.get_param('~chevron/threshold/window_size'),
        )
        self._chevron_pose_params: GetChevronPosesParams = GetChevronPosesParams(
            min_size=tuple(rospy.get_param('~chevron/pose/min_size')),
            max_size=tuple(rospy.get_param('~chevron/pose/max_size')),
            min_aspect_ratio=rospy.get_param('~chevron/pose/min_aspect_ratio'),
            max_aspect_ratio=rospy.get_param('~chevron/pose/max_aspect_ratio'),
            contour_approximation_factor=rospy.get_param('~chevron/pose/contour_approximation_factor'),
            angles=tuple(rospy.get_param('~chevron/pose/angles')),
            angle_match_tolerance=rospy.get_param('~chevron/pose/angle_match_tolerance'),
            depth_window_size=rospy.get_param('~chevron/pose/depth_window_size'),
        )

        self._lid_orange_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~lid/threshold/orange/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~lid/threshold/orange/local_thresholds')),
            window_size=rospy.get_param('~lid/threshold/orange/window_size'),
        )
        self._lid_purple_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~lid/threshold/purple/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~lid/threshold/purple/local_thresholds')),
            window_size=rospy.get_param('~lid/threshold/purple/window_size'),
        )
        self._lid_pose_params: GetLidPosesParams = GetLidPosesParams(
            orange_min_size=tuple(rospy.get_param('~lid/pose/orange/min_size')),
            orange_max_size=tuple(rospy.get_param('~lid/pose/orange/max_size')),
            orange_min_aspect_ratio=rospy.get_param('~lid/pose/orange/min_aspect_ratio'),
            orange_max_aspect_ratio=rospy.get_param('~lid/pose/orange/max_aspect_ratio'),
            orange_contour_approximation_factor=rospy.get_param('~lid/pose/orange/contour_approximation_factor'),
            purple_min_size=tuple(rospy.get_param('~lid/pose/purple/min_size')),
            purple_max_size=tuple(rospy.get_param('~lid/pose/purple/max_size')),
            purple_min_aspect_ratio=rospy.get_param('~lid/pose/purple/min_aspect_ratio'),
            purple_max_aspect_ratio=rospy.get_param('~lid/pose/purple/max_aspect_ratio'),
        )



def main():
    rospy.init_node('shape_detector')
    n = ShapeDetector()
    n.start()