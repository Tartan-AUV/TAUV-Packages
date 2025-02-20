#!/usr/bin/env python3

from sensor_msgs.msg import Image
import numpy as np
import cv2 as cv
import rospy
from cv_bridge import CvBridge
from tauv_common.srv import SetThresholdRequest, SetThresholdResponse

LOWER_THRESHOLD = np.array([0, 60, 60]) # TODO: figure out typing for threshold
UPPER_THRESHOLD = np.array([15, 255, 255])

class ThresholdNode(): 
    def __init__(self):
        print("running!")
        rospy.init_node('starter_threshold')

        self._lower_threshold : np.ndarray = LOWER_THRESHOLD
        self._upper_threshold : np.ndarray = UPPER_THRESHOLD

        self._cv_bridge : CvBridge = CvBridge()
        self._publisher : rospy.Publisher = rospy.Publisher('starter_task/thresholded_img', Image, queue_size=5)

        rospy.Subscriber('/kf/vehicle/oakd_bottom/stereo/right/image_color', Image, self.filter)
        rospy.Service('starter_task/set_threshold', SetThreshold, self.set_threshold)
        rospy.spin()

    def filter(self, data : Image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv_image, self._lower_threshold, self._upper_threshold)

        cv_result = cv.bitwise_and(cv_image, cv_image, mask=mask)

        result = self._cv_bridge.cv2_to_imgmsg(cv_result)

        self._publisher.publish(result)

    def set_threshold(self, req):
        self._lower_threshold = np.array([req.hue, req.saturation, req.value])
        return SetThresholdResponse(True)


def main():
    ThresholdNode()
