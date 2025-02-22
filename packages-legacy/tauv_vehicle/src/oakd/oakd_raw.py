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
        self._left = self._pipeline.create(depthai.node.MonoCamera)
        self._right = self._pipeline.create(depthai.node.MonoCamera)

        self._left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self._right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self._xout_color = self._pipeline.create(depthai.node.XLinkOut)
        self._xout_left = self._pipeline.create(depthai.node.XLinkOut)
        self._xout_right = self._pipeline.create(depthai.node.XLinkOut)

        self._xout_color.setStreamName('rgb')
        self._xout_left.setStreamName('left')
        self._xout_right.setStreamName('right')

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

        self._color.isp.link(self._xout_color.input)
        self._left.out.link(self._xout_left.input)
        self._right.out.link(self._xout_right.input)

        print("done setup")

        self._device = None
        while self._device is None and not rospy.is_shutdown():
            try:
                device_info = depthai.DeviceInfo(self._id)

                self._device = depthai.Device(self._pipeline, device_info)
            except Exception as e:
                rospy.logerr(f'OAKD device error: {e}')
                rospy.sleep(1.0)

        print("connected!")
        
        self._bridge = CvBridge()

        #estimate of ros system time offset compared to depthai clock
        depthai_time = depthai.Clock.now()
        self._time_offset = rospy.Time.now() - rospy.Time.from_sec(depthai_time.total_seconds())
        rospy.loginfo(f'Time offset: {self._time_offset}')

        self._color_pub = rospy.Publisher(f'vehicle/{self._frame}/color/image_raw', Image, queue_size=self._queue_size)
        self._left_pub = rospy.Publisher(f'vehicle/{self._frame}/left/image_raw', Image, queue_size=self._queue_size)
        self._right_pub = rospy.Publisher(f'vehicle/{self._frame}/right/image_raw', Image, queue_size=self._queue_size)

    def start(self):
        rgb_queue = self._device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
        left_queue = self._device.getOutputQueue(name='left', maxSize=1, blocking=False)
        right_queue = self._device.getOutputQueue(name='right', maxSize=1, blocking=False)

        while not rospy.is_shutdown():
            try:
                rgb = rgb_queue.tryGet()
                left = left_queue.tryGet()
                right = right_queue.tryGet()
            except Exception:
                continue

            if rgb is not None:
                print("rgb!")
                try:
                    img = self._bridge.cv2_to_imgmsg(rgb.getCvFrame()[:, :, ::-1], encoding='rgb8')
                    img.header.frame_id = self._frame
                    img.header.seq = rgb.getSequenceNum()
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(rgb.getTimestamp().total_seconds())
                    self._color_pub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo(f'OAKD frame error: {e}')

            if left is not None:
                print("left!")
                print(left.getCvFrame().shape)
                try:
                    img = self._bridge.cv2_to_imgmsg(left.getCvFrame(), encoding='mono8')
                    img.header.frame_id = self._frame
                    img.header.seq = left.getSequenceNum()
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(left.getTimestamp().total_seconds())
                    self._left_pub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo(f'OAKD frame error: {e}')

            if right is not None:
                print("right!")
                print(right.getCvFrame().shape)
                try:
                    img = self._bridge.cv2_to_imgmsg(right.getCvFrame(), encoding='mono8')
                    img.header.frame_id = self._frame
                    img.header.seq = right.getSequenceNum()
                    img.header.stamp = self._time_offset + rospy.Time.from_sec(right.getTimestamp().total_seconds())
                    self._right_pub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo(f'OAKD frame error: {e}')


    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame = rospy.get_param('~frame')
        self._id = rospy.get_param('~id')
        self._fps = 10
        self._queue_size = 10


def main():
    rospy.init_node('oakd')
    n = OAKDNode()
    n.start()