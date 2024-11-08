import rospy
import sys
import numpy as np
from PIL import Image
from math import floor

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header

class SonarMapPublisher:

    def __init__(self, map_path):
        self._map_path = map_path
        self._map_pub = rospy.Publisher('/sonar_map', OccupancyGrid, latch=True)
        print('init')

        self._map_msg = self._get_msg()

        self._map_pub.publish(self._map_msg)

    def _get_msg(self):
        img = Image.open(self._map_path)
        print(img.width)
        print(img.height)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = 'odom'
        grid_msg.info.map_load_time = rospy.Time.now()
        grid_msg.info.width = img.width
        grid_msg.info.height = img.height
        grid_msg.info.resolution = 21.0 / img.width
        grid_msg.info.origin.position = Vector3(-10.5 + 1.2, -10.5 + 2.5, 0)
        grid_msg.data = np.array(img.convert('L')).tolist()
        grid_msg.data = [0] * img.width * img.height

        gray_img = img.convert('L')

        for row in range(img.height):
            for col in range(img.width):

                grid_msg.data[row * img.width + col] = floor(gray_img.getpixel((row, col)) / 3)

        return grid_msg

    def _update(self, timer_event):
        print('publish')
        self._map_pub.publish(self._map_msg)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(1.0), self._update)
        rospy.spin()


def main():
    rospy.init_node('sonar_map_publisher')
    n = SonarMapPublisher(sys.argv[1])
    n.start()