#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
import numpy as np
from cv_bridge import CvBridge


class DisparityToDepthNode:
    def __init__(self):
        self._bridge = CvBridge()
        self._disparity_sub = rospy.Subscriber('disparity', DisparityImage, callback=self._handle_disparity, queue_size=10)
        self._depth_pub = rospy.Publisher('depth', Image, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_disparity(self, msg: DisparityImage):
        disp_img = self._bridge.imgmsg_to_cv2(msg.image)
        depth_img = np.where(disp_img > 0, (msg.T * msg.f) / disp_img, 0)
        depth_img = (1000 * depth_img).astype(np.uint16) # Integer mm
        depth_msg = self._bridge.cv2_to_imgmsg(depth_img, encoding='mono16')
        depth_msg.header = msg.header
        self._depth_pub.publish(depth_msg)


def main():
    rospy.init_node('disparity_to_depth')
    d = DisparityToDepthNode()
    d.start()
