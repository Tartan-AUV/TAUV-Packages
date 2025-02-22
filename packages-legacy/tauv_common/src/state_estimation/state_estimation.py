import rospy
import tf
import numpy as np
from typing import Any
from queue import PriorityQueue
from dataclasses import dataclass, field

from scipy.spatial.transform import Rotation
from tauv_util.types import tl, tm
from tauv_util.transforms import rpy_to_quat
from tauv_msgs.msg import Pose as PoseMsg, XsensImuData as ImuMsg, TeledyneDvlData as DvlMsg, FluidDepth as DepthMsg
from tauv_msgs.srv import SetPose, SetPoseRequest, SetPoseResponse
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Quaternion
from nav_msgs.msg import Odometry as OdometryMsg

from .ekf import EKF


@dataclass(order=True)
class StampedMsg:
    time: float
    msg: Any = field(compare=False)


class StateEstimation:

    def __init__(self):
        self._initialized = False

        self._tf_broadcaster: tf.TransformBroadcaster = tf.TransformBroadcaster()
        self._odom_tf_broadcaster: tf.TransformBroadcaster = tf.TransformBroadcaster()

        self._odom_world_pose: Pose = Pose()
        self._odom_world_pose.orientation = Quaternion(1.0, 0.0, 0.0, 0.0)

        self._set_pose_srv: rospy.Service = rospy.Service('set_pose', SetPose, self._handle_set_pose)

        self._pose_pub: rospy.Publisher = rospy.Publisher('pose', PoseMsg, queue_size=10)
        self._odom_pub: rospy.Publisher = rospy.Publisher('odom', OdometryMsg, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('/xsens_imu/data', ImuMsg, self._receive_msg)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('/teledyne_dvl/data', DvlMsg, self._receive_msg)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('depth', DepthMsg, self._receive_msg)

        self._dt: rospy.Duration = rospy.Duration.from_sec(1.0 / rospy.get_param('~frequency'))
        self._horizon_delay: rospy.Duration = rospy.Duration.from_sec(rospy.get_param('~horizon_delay'))

        dvl_offset = np.array(rospy.get_param('~dvl_offset'))
        process_covariance = np.array(rospy.get_param('~process_covariance'))

        self._imu_covariance = np.array(rospy.get_param('~imu_covariance'))
        self._dvl_covariance = np.array(rospy.get_param('~dvl_covariance'))
        self._depth_covariance = np.array(rospy.get_param('~depth_covariance'))

        self._ekf: EKF = EKF(dvl_offset, process_covariance)

        self._msg_queue = PriorityQueue()

        current_time = rospy.Time.now()
        self._last_horizon_time: rospy.Time = current_time - self._horizon_delay if current_time.to_sec() > self._horizon_delay.to_sec() else current_time

        self._initialized = True

    def _update(self, timer_event):
        if not self._initialized:
            return

        current_time = rospy.Time.now()

        if current_time.to_sec() < self._horizon_delay.to_sec():
            return

        horizon_time = current_time - self._horizon_delay

        while not self._msg_queue.empty():
            stamped_msg: StampedMsg = self._msg_queue.get()

            if stamped_msg.time < horizon_time.to_sec():
                msg = stamped_msg.msg

                if isinstance(msg, ImuMsg):
                    self._handle_imu(msg)
                elif isinstance(msg, DvlMsg):
                    self._handle_dvl(msg)
                elif isinstance(msg, DepthMsg):
                    self._handle_depth(msg)
            else:
                self._msg_queue.put(stamped_msg)
                break

        self._publish_state(current_time)

        self._last_horizon_time = horizon_time

    def _receive_msg(self, msg):
        if not self._initialized:
            return

        if msg.header.stamp.to_sec() < self._last_horizon_time.to_sec():
            return

        stamped_msg = StampedMsg(msg.header.stamp.to_sec(), msg)
        self._msg_queue.put(stamped_msg)

    def _handle_imu(self, msg: ImuMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        orientation = tl(msg.orientation)

        free_acceleration = tl(msg.free_acceleration)

        R = Rotation.from_euler('ZYX', np.flip(orientation)).inv()
        linear_acceleration = R.apply(free_acceleration)

        covariance = self._imu_covariance

        self._ekf.handle_imu_measurement(orientation, linear_acceleration, covariance, timestamp)

    def _handle_dvl(self, msg: DvlMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        if not msg.is_hr_velocity_valid:
            return

        velocity = tl(msg.hr_velocity)

        beam_std_dev = sum(msg.beam_standard_deviations) / 4.0

        covariance = self._dvl_covariance * np.array([beam_std_dev, beam_std_dev, beam_std_dev])

        self._ekf.handle_dvl_measurement(velocity, covariance, timestamp)

    def _handle_depth(self, msg: DepthMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        depth = np.array([msg.depth])
        covariance = self._depth_covariance

        self._ekf.handle_depth_measurement(depth, covariance, timestamp)

    def _publish_state(self, time: rospy.Time):
        try:
            position, velocity, acceleration, orientation, angular_velocity = self._ekf.get_state(time)
        except ValueError as e:
            return

        msg: PoseMsg = PoseMsg()
        msg.header = Header()
        msg.header.stamp = time
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = acceleration
        msg.orientation = orientation
        msg.angular_velocity = angular_velocity
        self._pose_pub.publish(msg)

        orientation_quat = rpy_to_quat(tl(orientation))

        odom_msg: OdometryMsg = OdometryMsg()
        odom_msg.header = Header()
        odom_msg.header.stamp = time
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'vehicle'
        odom_msg.pose = PoseWithCovariance()
        odom_msg.pose.pose = Pose(
            position=position,
            orientation=orientation_quat,
        )
        odom_msg.twist = TwistWithCovariance()
        odom_msg.twist.twist = Twist(
            linear=velocity,
            angular=angular_velocity
        )
        self._odom_pub.publish(odom_msg)

        self._tf_broadcaster.sendTransform(
            translation=(position.x, position.y, position.z),
            rotation=(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w),
            time=time,
            child='vehicle',
            parent='odom',
        )
        self._send_odom_transform()

    def _handle_set_pose(self, req: SetPoseRequest):
        self._odom_world_pose = req.pose
        self._send_odom_transform()
        return SetPoseResponse(True)

    def _send_odom_transform(self):
        pos = self._odom_world_pose.position
        rot = self._odom_world_pose.orientation

        self._odom_tf_broadcaster.sendTransform(
            translation=(pos.x, pos.y, pos.z),
            rotation=(rot.x, rot.y, rot.z, rot.w),
            time=rospy.Time.now(),
            child='odom',
            parent='world',
        )

    def start(self):
        self._send_odom_transform()
        rospy.Timer(self._dt, self._update)
        rospy.spin()


def main():
    rospy.init_node('state_estimation')
    n = StateEstimation()
    n.start()
