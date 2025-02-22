import rospy
import numpy as np
import sensor_msgs.msg
from typing import Optional
import threading
import math
import scipy as sp
import sys
import collections
from PIL import Image
import cv2
from cv_bridge import CvBridge
from skimage.restoration import denoise_bilateral
from skimage.morphology import closing, disk, square
from skimage import exposure
from vision.debluer.utility import NUCE


class DeluberParams:
    def __init__(self) -> None:
        self.f = 2.0
        self.l = 0.5
        self.p = 0.01
        self.min_depth = 0.1
        self.spread_data_fraction = 0.01
        self.additive_depth = 2.0
        self.multiply_depth = 10.0

class Debluer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lock.acquire()
        self.frame_id = rospy.get_param('~frame_id')
        self.cv_bridge = CvBridge()
        self.img_sub = rospy.Subscriber(f'vehicle/{self.frame_id}/color/image_raw', sensor_msgs.msg.Image, self.handle_img)
        self.pub = rospy.Publisher(f'vehicle/{self.frame_id}/color/debluer', sensor_msgs.msg.Image)
        self.lock.release()

        self.deluber_params = DeluberParams()

    def start(self):
        rospy.spin()

    def handle_img(self, msg):
        self.lock.acquire()

        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img.thumbnail(img.size, Image.LANCZOS)
        # img_adapteq = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
        # result = np.round(img_adapteq * 255.0).astype(np.uint8)
        # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = NUCE(img)
        result_msg = self.cv_bridge.cv2_to_imgmsg(result, encoding="bgr8")
        result_msg.header = msg.header
        self.pub.publish(result_msg)

        self.lock.release()
    
def main():
    rospy.init_node('debluer')
    n = Debluer()
    n.start()
