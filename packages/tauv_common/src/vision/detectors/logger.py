#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import FeatureDetections, FeatureDetection
from tauv_msgs.srv import MapFind, MapFindClosest
from visualization_msgs.msg import MarkerArray, Marker
from tauv_util.transforms import rpy_to_quat
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

def getColor(tag, trans=False):
    color = [0,0,0,1]

    if(tag=="phone"):
        color = [1,0,0,1]

    if(tag=="badge"):
        color = [0,1,0,1]

    if(tag=="notebook"):
        color = [0,0,1,1]

    if(tag=="gate"):
        color = [1,1,0,1]

    if (tag == "circle"):
        color = [0, 1, 1, 1]

    if (tag == "chevron"):
        color = [0, 0.5, 1, 1]

    if(not trans):
        color[3] = 0.75

    return color

def makeMarker(tag, id, detection, color, scale = 0.5, shape = Marker.SPHERE, time = rospy.Duration(1.0)):
    marker = Marker()
    marker.header.frame_id = "kf/odom" # FIX THIS
    marker.header.stamp = rospy.Time()
    marker.ns = tag
    marker.id = id
    marker.type = shape
    marker.action = Marker.ADD
    marker.pose.position.x = detection.position.x
    marker.pose.position.y = detection.position.y
    marker.pose.position.z = detection.position.z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.a = color[3]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.lifetime = time
    
    return marker

def makeCircleMarker(id, detection):
    marker = Marker()
    marker.header.frame_id = "kf/odom" # FIX THIS
    marker.header.stamp = rospy.Time()
    marker.ns = 'circle'
    marker.id = id
    marker.type = Marker.ARROW
    marker.pose.position.x = detection.position.x
    marker.pose.position.y = detection.position.y
    marker.pose.position.z = detection.position.z
    marker.pose.orientation = rpy_to_quat(np.array([
        detection.orientation.x,
        detection.orientation.y,
        detection.orientation.z,
    ]))
    marker.scale.x = 1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.r = 1
    marker.color.g = 1
    marker.color.b = 0
    marker.color.a = 1
    marker.lifetime = rospy.Duration(1.0)

    return marker


class Logger():
    def __init__(self):
        rospy.init_node('logger')

        self.ind = 10000
        self.viz = rospy.Publisher("global_map/visualization_marker_array", MarkerArray, queue_size=100)
        self.find = rospy.ServiceProxy("global_map/find", MapFind)
        self.bridge = CvBridge()

        rospy.Subscriber("global_map/feature_detections", FeatureDetections,
                                                self.publish)

        # ???
        # rospy.Subscriber("global_map/find", FeatureDetections,
        #                                 self.visualize)

        rospy.wait_for_service("global_map/find")

        #rospy.Subscriber("/oakd/oakd_front/depth_map", Image, self.depth)
        #rospy.Subscriber("/oakd/oakd_front/color_image", Image, self.color)

        rospy.Timer(rospy.Duration(0.5), self.visualize)

    def color(self, data):
        print("COLOR")
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        print(data.header)
        print(cv_image)

    def depth(self, data):
        print("BW")
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono16')
        print(data.header)
        print(cv_image)

    def visualize(self, time):
        #buckets = bucketList.bucket_list
        buckets1 = self.find("badge").detections
        buckets2 = self.find("phone").detections
        buckets3 = self.find("notebook").detections
        buckets4 = self.find("gate").detections
        buckets5 = self.find("circle").detections
        buckets6 = self.find("chevron").detections

        markers = []
        for ind in range(len(buckets1)):
            det = buckets1[ind]
            markers.append(makeMarker("badge", ind, det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets2)):
            det = buckets2[ind]
            markers.append(makeMarker("phone", ind+len(buckets1), det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets3)):
            det = buckets3[ind]
            markers.append(makeMarker("notebook", ind+len(buckets1)+len(buckets2), det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets4)):
            det = buckets4[ind]
            markers.append(makeMarker("gate", ind+len(buckets1)+len(buckets2)+len(buckets3), det, getColor(det.tag), 1, Marker.CUBE, rospy.Duration(5.0)))

        for ind in range(len(buckets5)):
            det = buckets5[ind]
            markers.append(makeCircleMarker(ind+len(buckets1)+len(buckets2)+len(buckets3)+len(buckets4), det))

        for ind in range(len(buckets6)):
            det = buckets6[ind]
            markers.append(makeCircleMarker(ind+len(buckets1)+len(buckets2)+len(buckets3)+len(buckets4)+len(buckets5), det))

        OBJ = MarkerArray()
        OBJ.markers = markers

        self.viz.publish(OBJ)

    def publish(self, objects):
        detections = objects.detections
        markers = []
        for detection in detections:
            markers.append(makeMarker(detection.tag, self.ind, detection, getColor(detection.tag, True), scale=0.05))
            self.ind += 1

        markersPub = MarkerArray()
        markersPub.markers = markers
        self.viz.publish(markersPub)

def main():
    s = Logger()
    rospy.spin()