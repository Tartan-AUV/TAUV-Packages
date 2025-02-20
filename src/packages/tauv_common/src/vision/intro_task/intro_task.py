#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class find_red:
    def __init__(self):
        # "/kf/vehicle/oakd_bottom/color/camera_info"
        self.c = CvBridge()
        pub_name = "intro_task/new_image"
        # sub_name = "/kf/vehicle/oakd_bottom/color/image_raw"
        sub_name = "/kf/vehicle/oakd_bottom/stereo/right/image_color"

        self.pub = rospy.Publisher(pub_name, Image, queue_size = 10)
        self.imageSub = rospy.Subscriber(sub_name, Image, self.image_callback)
        self.img = Image()
        self.mask = Image()

    def image_callback(self, msg):
        self.img = self.c.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # img_thresh = cv2.threshold(self.img, low = (100, 0, 0))
        # self.img = np.where(self.img[self.img[:,:,0] > 50, self.img, 0])
        # mask = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY_INV)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])
        self.mask = cv2.inRange(hsv, lower_red, upper_red)
        img = cv2.bitwise_and(self.img, self.img, mask=self.mask)
        self.img = self.c.cv2_to_imgmsg(img)
        
        self.pub.publish(self.img)
        # self.pub.publish(self.mask)



if __name__ == '__main__':
    node_name = 'intro_task'
    rospy.init_node(node_name)
    find_red()
    rospy.spin
