#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
import numpy as np
from cv_bridge import CvBridge


class Mono16Node:
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber('input_depth', Image, callback=self._handle_image, queue_size=10)
        self._depth_pub = rospy.Publisher('output_depth', Image, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_image(self, msg: Image):
        depth_img = self._bridge.imgmsg_to_cv2(msg)
        depth_img = np.nan_to_num(1000 * depth_img).astype(np.uint16) # Integer mm
        depth_msg = self._bridge.cv2_to_imgmsg(depth_img, encoding='mono16')
        depth_msg.header = msg.header
        self._depth_pub.publish(depth_msg)


def main():
    rospy.init_node('f1c_to_mono16')
    d = Mono16Node()
    d.start()
