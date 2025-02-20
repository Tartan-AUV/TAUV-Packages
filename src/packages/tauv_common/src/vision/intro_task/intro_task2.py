#!/usr/bin/env python3

import rospy
import rviz
import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

class handle_img:
    def __init__(self):
        self.c = CvBridge()
        
        sub_name = "/kf/vehicle/oakd_bottom/color/new_image"

        self.imageSub = rospy.Subscriber(sub_name, Image, self.image_callback)
        self.img = Image()

    def image_callback(self, msg):
        self.img = msg
        # self.frame



if __name__ == '__main__':
    rospy.init_node('new image topic')
    handle_img()
    rospy.spin