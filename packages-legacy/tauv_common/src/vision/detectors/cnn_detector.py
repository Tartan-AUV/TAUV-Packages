# cnn_detector
#
# This node is for CNN object detection
#
#
# Author: Advaith Sethuraman 2020

#!/usr/bin/env python
from __future__ import division
import rospy
import tf
import tf_conversions
import numpy as np
import itertools
import cv2
from cv_bridge import CvBridge
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import time
import datetime
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from sensor_msgs.msg import Imu, Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import *
from jsk_recognition_msgs.msg import BoundingBox
from nav_msgs.msg import Odometry
from tf.transformations import *
from std_msgs.msg import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList
from tauv_common.srv import RegisterObjectDetections
from scipy.spatial.transform import Rotation as R
import time


class Dummy_Detector():
    def __init__(self):
        self.detector_id = "yolov3"

        #CNN params
        self.weights = "./yolov3.weights"
        self.config = "./yolov3.cfg"
        self.classes_list = "./yolov3.txt"
        if rospy.has_param("detectors/" + self.detector_id + "/config_path"):
            self.config = str(rospy.get_param("detectors/" + self.detector_id + "/config_path"))
        rospy.loginfo("[CNN Detector]: Loading CNN Config from: %s..." % self.config)
        if rospy.has_param("detectors/" + self.detector_id + "/weights_path"):
            self.weights = str(rospy.get_param("detectors/" + self.detector_id + "/weights_path"))
        rospy.loginfo("[CNN Detector]: Loading CNN Weights from: %s..." % self.weights)
        if rospy.has_param("detectors/" + self.detector_id + "/classes_path"):
            self.classes_list = str(rospy.get_param("detectors/" + self.detector_id + "/classes_path"))
        rospy.loginfo("[CNN Detector]: Loading CNN Classes from: %s..." % self.classes_list)

        self.conf_threshold = 0.8
        self.nms_threshold = 0.4
        self.img_size = 416
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Darknet(self.config, img_size=self.img_size).to(self.device)
        self.net.load_darknet_weights(self.weights)
        self.net.eval()
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.arrow_list = []
        self.classes = self.prepare_classes()
        self.stereo_proc = cv2.StereoBM_create(numDisparities=16, blockSize=33)
        self.tf = tf.TransformListener()
        self.cv_bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.orb = cv2.ORB_create(300)
        self.focal_length = 0.0
        self.stereo_left = Image()
        self.stereo_right = []
        self.disparity = DisparityImage()
        self.baseline = 0.03
        self.left_camera_info = CameraInfo()
        self.left_img_flag = False
        self.disp_img_flag = False
        self.left_info_flag = False
        self.rate = rospy.Rate(1)
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        self.marker_id = 0

        rospy.wait_for_service("detector_bucket/register_object_detection")
        self.registration_service = rospy.ServiceProxy("detector_bucket/register_object_detection", RegisterObjectDetections)
        self.left_stream = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)
        self.disparity_stream = rospy.Subscriber("/vision/front/disparity", DisparityImage, self.disparity_callback)
        self.left_camera_info = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_info", CameraInfo, self.camera_info_callback)
        self.left_camera_detections = rospy.Publisher("cnn_detections", Image, queue_size=10)

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = [0, 0, 255]
        cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)
        cv2.putText(img, label, (int(x-10), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def load_model(self):
        return cv2.dnn.readNet(self.weights, self.config)

    def prepare_classes(self):
        classes = None
        with open(self.classes_list, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def classify(self, image):
        now = rospy.Time.now()
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        pad_x = int(max(Height - Width, 0) // 2)
        pad_y = int(max(Width - Height, 0) // 2)
        padded_img = cv2.copyMakeBorder(norm_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_img = cv2.resize(padded_img, (self.img_size, self.img_size))
        popped_tensor = torch.unsqueeze(torch.tensor(resized_img), dim=0).permute(0, 3, 1, 2)
        input_img = Variable(popped_tensor.type(self.Tensor))
        with torch.no_grad():
            detections = self.net(input_img)
            detections = non_max_suppression(detections, self.conf_threshold, self.nms_threshold)

        class_ids = []
        confidences = []
        boxes = []

        for det in detections:
            if det != None:
                outs = rescale_boxes(det, self.img_size, image.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in outs:
                    if conf > self.conf_threshold:
                        w = x2 - x1
                        h = y2 - y1
                        x = x1
                        y = y1
                        class_ids.append(int(cls_pred))
                        confidences.append(float(conf))
                        boxes.append([x, y, w, h])

        final_detections = []
        for bi, box in enumerate(boxes):
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            class_id = class_ids[bi]
            confidence = confidences[bi]
            if not np.any(np.isnan(np.array([x, y, w, h]))):
                final_detections.append([class_id, (int(x), int(y), int(w), int(h))])
                self.draw_bounding_box(image, class_id, confidence, round(x), round(y), round(x+w), round(y+h))

        image = cv2.putText(image, "[%s] : %f" % (self.detector_id, now.to_sec()), \
                             (Width - 170, Height - 15), cv2.FONT_HERSHEY_SIMPLEX, \
                             .5, (0, 0, 255), 2)
        self.left_camera_detections.publish(self.cv_bridge.cv2_to_imgmsg(image))
        return final_detections, now

    def camera_info_callback(self, msg):
        self.left_camera_info = msg
        self.left_info_flag = True

    def disparity_callback(self, msg):
        self.disp_img_flag = True
        self.disparity = msg
        self.baseline = msg.T

    def left_callback(self, msg):
        self.stereo_left = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.left_img_flag = True

    def query_depth_map(self, map, pixel):
        return map[pixel[:,1], pixel[:,0]]

    def query_depth_map_rectangle(self, map, bbox_detection):
        x, y, w, h = bbox_detection[1]
        return map[y:y+h, x:x+w].reshape(-1, 1)

    def depth_from_disparity(self, msg):
        self.focal_length = msg.f
        self.baseline = msg.T
        disparity_image = self.cv_bridge.imgmsg_to_cv2(msg.image, "passthrough")
        return self.focal_length * self.baseline / disparity_image

    def vector_to_detection_centroid(self, bbox_detection):
        x, y, w, h = bbox_detection[1]
        disp_map = self.cv_bridge.imgmsg_to_cv2(self.disparity.image, "passthrough")
        x_cnt = int(x+w/2)
        y_cnt = int(y+h/2)
        d = np.mean(self.query_depth_map_rectangle(disp_map, bbox_detection))
        centroid_2d = np.asmatrix([x_cnt, y_cnt, d, 1]).T
        K = np.asarray(self.left_camera_info.K).reshape((3, 3))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Q = np.asmatrix([[1, 0, 0, -cx],
                         [0, 1, 0, -cy],
                         [0, 0, 0, fx],
                         [0, 0, -1.0/0.03, 0]])
        centroid_3d = Q * centroid_2d
        centroid_3d /= centroid_3d[3]
        return centroid_3d[0:3]

    def prepare_detection_registration(self, centroid, det, now):
        obj_det = BucketDetection()
        obj_det.image = self.cv_bridge.cv2_to_imgmsg(self.stereo_left, "bgr8")
        obj_det.tag = str("object_tags/" + self.classes[det[0]])
        bbox_dims = np.asarray([1.0, 1.0, 1.0])
        if rospy.has_param("object_tags/" + self.classes[det[0]] + "/dimensions"):
            bbox_dims = np.asarray(rospy.get_param("object_tags/" + self.classes[det[0]] + "/dimensions")).astype(float)
        bbox_3d = BoundingBox()
        bbox_3d.dimensions = Vector3(bbox_dims[0], bbox_dims[1], bbox_dims[2])
        bbox_pose = Pose()
        x, y, z = list((np.squeeze(centroid)).T)
        obj_det.position = Point(x, y, z)
        bbox_pose.position = Point(x, y, z)
        bbox_3d.pose = bbox_pose
        bbox_header = Header()
        bbox_header.frame_id = "duo3d_optical_link_front"
        bbox_header.stamp = now
        bbox_3d.header = bbox_header
        obj_det.bbox_3d = bbox_3d
        obj_det.header = Header()
        obj_det.header.frame_id = bbox_header.frame_id
        obj_det.header.stamp = now
        return obj_det

    def spin(self, event):
        if(self.left_img_flag and self.disp_img_flag and self.left_info_flag):
            self.left_img_flag = False
            self.disp_img_flag = False
            self.left_info_flag = False
            detections, now = self.classify(self.stereo_left)
            det_packet = []
            for det in detections:
                feature_centroid = self.vector_to_detection_centroid(det)
                det_packet.append(self.prepare_detection_registration(feature_centroid, det, now))
            success = self.registration_service(det_packet, self.detector_id)


def main():
    rospy.init_node("cnn_detector")
    dummy_detector = Dummy_Detector()
    rospy.spin()