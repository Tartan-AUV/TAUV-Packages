import rospy
import numpy as np
from tauv_msgs.msg import XsensImuData as ImuMsg, FluidDepth as DepthMsg, NavigationState


class AlbatrossStateEstimation:

    def __init__(self):
        self._dt: float = 0.02
        self._navigation_state_pub: rospy.Publisher = rospy.Publisher('gnc/navigation_state', NavigationState, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/data_raw', ImuMsg, self._handle_imu)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('vehicle/arduino/depth', DepthMsg, self._handle_depth)

        self._imu_msg = None
        self._depth_msg = None

    def _update(self, timer_event):
        nav_state = NavigationState()

        if self._imu_msg is not None:
            nav_state.orientation.x = self._imu_msg.orientation.x
            nav_state.orientation.y = self._imu_msg.orientation.y
            nav_state.orientation.z = self._imu_msg.orientation.z
            nav_state.euler_velocity.x = self._imu_msg.rate_of_turn.x
            nav_state.euler_velocity.y = self._imu_msg.rate_of_turn.y
            nav_state.euler_velocity.z = self._imu_msg.rate_of_turn.z

        if self._depth_msg is not None:
            nav_state.position.z = self._depth_msg.depth

        self._navigation_state_pub.publish(nav_state)

    def start(self):
        rospy.Timer(rospy.Duration(self._dt), self._update)
        rospy.spin()

    def _handle_imu(self, msg: ImuMsg):
        self._imu_msg = msg

    def _handle_depth(self, msg: DepthMsg):
        self._depth_msg = msg


def main():
    rospy.init_node('albatross_state_estimation')
    n = AlbatrossStateEstimation()
    n.start()
