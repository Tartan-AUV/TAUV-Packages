#!/usr/bin/env python3

import logging
from typing import Dict

import rospy
import depthai
from Cython.Compiler.Naming import self_cname
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

# node to create oakd ros nodes from oakd api
# publishes depth map and color image

logger = logging.getLogger(__name__)

class OAKDNode:
    def __init__(self):
        self._load_config()

        self._pipeline = depthai.Pipeline()

        self._color = self._pipeline.create(depthai.node.ColorCamera)
        self._depth = self._pipeline.create(depthai.node.StereoDepth)
        self._left = self._pipeline.create(depthai.node.MonoCamera)
        self._right = self._pipeline.create(depthai.node.MonoCamera)

        self._xout_color = self._pipeline.create(depthai.node.XLinkOut)
        self._xout_depth = self._pipeline.create(depthai.node.XLinkOut)

        self._xout_color.setStreamName('color')
        self._xout_depth.setStreamName('depth')

        self._left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P)
        self._left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self._left.setFps(self._fps)
        self._left.initialControl.setManualFocus(255)

        self._right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P)
        self._right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
        self._right.setFps(self._fps)
        self._right.initialControl.setManualFocus(255)

        self._color.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self._color.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_12_MP)
        self._color.setInterleaved(False)
        self._color.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
        self._color.setFps(self._fps)
        self._color.initialControl.setManualFocus(255)

        if self._color_downsample_res:
            self._color_manip = self._pipeline.create(depthai.node.ImageManip)
            self._color_manip.setResize(*self._color_downsample_res)
            self._color_manip.setMaxOutputFrameSize(self._color_downsample_res[0] * self._color_downsample_res[1] * 3)

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
        if self._color_downsample_res:
            self._color.isp.link(self._color_manip.inputImage)
            self._color_manip.out.link(self._xout_color.input)
        else:
            self._color.isp.link(self._xout_color.input)

        if self._publish_mono:
            self._xout_left = self._pipeline.create(depthai.node.XLinkOut)
            self._xout_right = self._pipeline.create(depthai.node.XLinkOut)

            self._xout_left.setStreamName('left')
            self._xout_right.setStreamName('right')

            self._left.out.link(self._xout_left.input)
            self._right.out.link(self._xout_right.input)

        self._device = None
        while self._device is None and not rospy.is_shutdown():
            try:
                device_info = depthai.DeviceInfo(self._id)

                self._device = depthai.Device(self._pipeline, device_info)
            except Exception as e:
                rospy.logerr(f'OAKD device error: {e}')
                rospy.sleep(1.0)

        self._calibration = self._device.readCalibration()

        self._camera_info = dict()

        self._camera_info['depth'] = CameraInfo()
        self._camera_info['depth'].K = np.ndarray.flatten(np.array(self._calibration.getCameraIntrinsics(depthai.CameraBoardSocket.LEFT)))
        self._camera_info['depth'].distortion_model = 'rational_polynomial'
        self._camera_info['depth'].D = np.array(self._calibration.getDistortionCoefficients(depthai.CameraBoardSocket.LEFT))
        self._camera_info['color'] = CameraInfo()
        self._camera_info['color'].K = np.ndarray.flatten(np.array(self._calibration.getCameraIntrinsics(depthai.CameraBoardSocket.RGB, resizeWidth=640, resizeHeight=360)))
        self._camera_info['color'].distortion_model = 'rational_polynomial'
        self._camera_info['color'].D = np.array(self._calibration.getDistortionCoefficients(depthai.CameraBoardSocket.RGB))
        self._camera_info['left'] = self._camera_info['depth']
        self._camera_info['right'] = CameraInfo()
        self._camera_info['right'].K = np.ndarray.flatten(np.array(self._calibration.getCameraIntrinsics(depthai.CameraBoardSocket.RIGHT)))
        self._camera_info['right'].distortion_model = 'rational_polynomial'
        self._camera_info['right'].D = np.array(self._calibration.getDistortionCoefficients(depthai.CameraBoardSocket.RIGHT))

        self._bridge = CvBridge()

        #estimate of ros system time offset compared to depthai clock
        depthai_time = depthai.Clock.now()
        self._time_offset = rospy.Time.now() - rospy.Time.from_sec(depthai_time.total_seconds())
        rospy.loginfo(f'Time offset: {self._time_offset}')

        self._topics = ['left', 'right', 'color', 'depth'] if self._publish_mono else ['color', 'depth']
        self._image_pubs = dict()
        self._info_pubs = dict()
        for topic in self._topics:
            self._image_pubs[topic] = rospy.Publisher(f'vehicle/{self._frame}/{topic}/image_raw', Image, queue_size=self._queue_size)
            self._info_pubs[topic] = rospy.Publisher(f'vehicle/{self._frame}/{topic}/camera_info', CameraInfo, queue_size=1, latch=True)

    def start(self):
        queues = dict()
        for topic in self._topics:
            queues[topic] = self._device.getOutputQueue(topic, self._queue_size, blocking=True)

        messages: Dict[str, depthai.ImgFrame] = dict()

        conversions = {
            'left': lambda frame: self._bridge.cv2_to_imgmsg(frame.getCvFrame(), encoding='mono8'),
            'right': lambda frame: self._bridge.cv2_to_imgmsg(frame.getCvFrame(), encoding='mono8'),
            'color': lambda frame: self._bridge.cv2_to_imgmsg(frame.getCvFrame()[:, :, ::-1], encoding='rgb8'),
            'depth': lambda frame: self._bridge.cv2_to_imgmsg(frame.getCvFrame(), encoding='mono16')
        }

        for topic in self._topics:
            self._info_pubs[topic].publish(self._camera_info['depth'] if topic == 'depth' else self._camera_info['color'])

        while not rospy.is_shutdown():
            try:
                for name, queue in queues.items():
                    messages[name] = queue.get()
            except Exception:
                continue

            seq_num = messages['color'].getSequenceNum()
            sync_needed = False
            for msg in messages.values():
                print(f'{msg.getSequenceNum()}', end=' ')
                if msg.getSequenceNum() != seq_num:
                    sync_needed = True
                    break
            print()
            if sync_needed:
                self._sync(queues)

            try:
                for topic, message in messages.items():
                    img = conversions[topic](message)
                    img.header.frame_id = self._frame
                    img.header.seq = seq_num
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(message.getTimestamp().total_seconds())
                    self._image_pubs[topic].publish(img)

            except CvBridgeError as e:
                rospy.loginfo(f'OAKD frame error: {e}')


    def _sync(self, queues: Dict[str, depthai.DataOutputQueue]):
        rospy.loginfo("OAKD streams out of sync, syncing...")

        messages: Dict[str, depthai.ImgFrame] = dict()

        for topic, queue in queues.items():
            messages[topic] = queue.get()
            rospy.logwarn(f'OAKD stream {topic} at {messages[topic].getSequenceNum()}')

        max_seq_num = max([frame.getSequenceNum() for frame in messages.values()])

        synced_topics = []
        for topic, msg in messages.items():
            if msg.getSequenceNum() == max_seq_num:
                synced_topics.append(topic)

        counter = 0
        while len(synced_topics) < len(messages) and counter < self._queue_size:
            for topic, queue in queues.items():
                if topic in synced_topics:
                    continue
                msg = queue.get()
                if msg.getSequenceNum() == max_seq_num:
                    rospy.loginfo(f'OAKD synced {topic} at {msg.getSequenceNum()}')
                    synced_topics.append(topic)
            counter += 1

        if counter == self._queue_size:
            rospy.logerr('OAKD sync error: timeout')
            exit(1)
        else:
            rospy.loginfo('OAKD streams synced')

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame = rospy.get_param('~frame')
        self._id = rospy.get_param('~id')
        self._postprocess_depth = True
        self._publish_mono = True
        self._color_downsample_res = None
        self._fps = 1
        self._queue_size = 10

def main():
    rospy.init_node('oakd')
    n = OAKDNode()
    n.start()