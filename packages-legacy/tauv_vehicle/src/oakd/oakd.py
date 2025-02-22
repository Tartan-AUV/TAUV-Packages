#!/usr/bin/env python3

import rospy
import depthai
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

# node to create oakd ros nodes from oakd api
# publishes depth map and color image

class OAKDNode:
    def __init__(self):
        self._load_config()

        self._pipeline = depthai.Pipeline()

        self._color = self._pipeline.create(depthai.node.ColorCamera)
        self._depth = self._pipeline.create(depthai.node.StereoDepth)
        self._left = self._pipeline.create(depthai.node.MonoCamera)
        self._right = self._pipeline.create(depthai.node.MonoCamera)

        self._left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self._right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self._xout_color = self._pipeline.create(depthai.node.XLinkOut)
        self._xout_depth = self._pipeline.create(depthai.node.XLinkOut)

        self._xout_color.setStreamName('rgb')
        self._xout_depth.setStreamName('depth')

        self._left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
        self._left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self._left.setFps(self._fps)

        self._right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
        self._right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
        self._right.setFps(self._fps)

        self._color.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self._color.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self._color.setInterleaved(False)
        self._color.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
        self._color.setFps(self._fps)

        self._color_manip = self._pipeline.create(depthai.node.ImageManip)
        self._color_manip.setResize(640, 360)
        self._color_manip.setMaxOutputFrameSize(640 * 360 * 3)

        self._depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self._depth.setLeftRightCheck(True)
        self._depth.setExtendedDisparity(False)
        self._depth.setSubpixel(False)
        self._depth.setOutputSize(640, 360)
        self._depth.setDepthAlign(depthai.CameraBoardSocket.RGB)
        self._depth.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)

        if self._postprocess_depth:
            config = self._depth.initialConfig.get()
            config.postProcessing.speckleFilter.enable = True
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = True
            config.postProcessing.spatialFilter.enable = True
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            config.postProcessing.thresholdFilter.minRange = 400
            config.postProcessing.thresholdFilter.maxRange = 15000
            config.postProcessing.decimationFilter.decimationFactor = 1
            self._depth.initialConfig.set(config)

        self._left.out.link(self._depth.left)
        self._right.out.link(self._depth.right)
        self._depth.depth.link(self._xout_depth.input)
        self._color.isp.link(self._color_manip.inputImage)
        self._color_manip.out.link(self._xout_color.input)

        self._device = None
        while self._device is None and not rospy.is_shutdown():
            try:
                device_info = depthai.DeviceInfo(self._id)

                self._device = depthai.Device(self._pipeline, device_info)
            except Exception as e:
                rospy.logerr(f'OAKD device error: {e}')
                rospy.sleep(1.0)
        
        self._calibration = self._device.readCalibration()
        self._depth_info = CameraInfo()
        self._depth_info.K = np.ndarray.flatten(np.array(self._calibration.getCameraIntrinsics(depthai.CameraBoardSocket.LEFT)))
        self._depth_info.distortion_model = 'rational_polynomial'
        self._depth_info.D = np.array(self._calibration.getDistortionCoefficients(depthai.CameraBoardSocket.LEFT))
        self._color_info = CameraInfo()
        self._color_info.K = np.ndarray.flatten(np.array(self._calibration.getCameraIntrinsics(depthai.CameraBoardSocket.RGB, resizeWidth=640, resizeHeight=360)))
        self._color_info.distortion_model = 'rational_polynomial'
        self._color_info.D = np.array(self._calibration.getDistortionCoefficients(depthai.CameraBoardSocket.RGB))

        self._bridge = CvBridge()

        #estimate of ros system time offset compared to depthai clock
        depthai_time = depthai.Clock.now()
        self._time_offset = rospy.Time.now() - rospy.Time.from_sec(depthai_time.total_seconds())
        rospy.loginfo(f'Time offset: {self._time_offset}')

        self._depth_pub = rospy.Publisher(f'vehicle/{self._frame}/depth/image_raw', Image, queue_size=self._queue_size)
        self._color_pub = rospy.Publisher(f'vehicle/{self._frame}/color/image_raw', Image, queue_size=self._queue_size)
        self._depth_info_pub = rospy.Publisher(f'vehicle/{self._frame}/depth/camera_info', CameraInfo, queue_size=1, latch=True)
        self._color_info_pub = rospy.Publisher(f'vehicle/{self._frame}/color/camera_info', CameraInfo, queue_size=1, latch=True)

    def start(self):
        rgb_queue = self._device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
        depth_queue = self._device.getOutputQueue(name='depth', maxSize=1, blocking=False)

        while not rospy.is_shutdown():
            try:
                rgb = rgb_queue.tryGet()
                depth = depth_queue.tryGet()
            except Exception:
                continue

            self._depth_info_pub.publish(self._depth_info)
            self._color_info_pub.publish(self._color_info)

            if rgb is not None:
                try:
                    img = self._bridge.cv2_to_imgmsg(rgb.getCvFrame()[:, :, ::-1], encoding='rgb8')
                    img.header.frame_id = self._frame
                    img.header.seq = rgb.getSequenceNum()
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(rgb.getTimestamp().total_seconds())
                    self._color_pub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo(f'OAKD frame error: {e}')

            if depth is not None:
                try:
                    img = self._bridge.cv2_to_imgmsg(depth.getCvFrame(), encoding='mono16')
                    img.header.frame_id = self._frame
                    img.header.seq = depth.getSequenceNum()
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(depth.getTimestamp().total_seconds())
                    self._depth_pub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo(f'OAKD frame error: {e}')

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame = rospy.get_param('~frame')
        self._id = rospy.get_param('~id')
        self._postprocess_depth = True
        self._fps = 10
        self._queue_size = 10

def main():
    rospy.init_node('oakd')
    n = OAKDNode()
    n.start()