# motion_utils.y
#
# Simple python library intended to expose motion control of the AUV to mission scripts.
# Only one MotionUtils class should be instantiated at a time. (TODO: make this a python singleton)
# MotionUtils is intended to be used in conjunction with the mpc_trajectory_follower planner.
#
#
# To use: one MotionUtils instance should be created at sub startup, then passed to any mission scripts
# from the mission manager node.
# The MotionUtils class will automatically provide updates to the trajectory follower.
# Use update_trajectory to update the reference trajectory object to be used by motion_utils.
#
#

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Twist, Quaternion, Vector3
from tauv_msgs.srv import GetTrajectory, GetTrajectoryRequest, GetTrajectoryResponse
from std_srvs.srv import SetBool, SetBoolRequest, Trigger
from tauv_util.transforms import twist_body_to_world, quat_to_rpy
from nav_msgs.msg import Path, Odometry as OdometryMsg
from motion.trajectories import Trajectory, TrajectoryStatus
from tauv_util.types import tl, tm
from scipy.spatial.transform import Rotation
from motion.trajectories.fixed_linear_trajectory import LinearTrajectory
import typing
from tauv_msgs.msg import TrajPoint
from tauv_msgs.srv import SetPose, SetPoseRequest, SetPoseResponse
from std_msgs.msg import Float64
from motion.trajectories.pyscurve.pyscurve.trajectory import PlanningError
from motion.trajectories.hold_pos import HoldPos

class MotionUtils:
    def __init__(self):

        self.initialized = False
        self.traj = None

        self.pose = None
        self.twist = None

        self._tf_namespace = rospy.get_param('tf_namespace')

        self._traj_service = rospy.Service('gnc/get_trajectory', GetTrajectory, self._handle_get_traj)

        self._arm_srv = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

        self._path_pub = rospy.Publisher('gnc/path', Path, queue_size=10)
        self._target_pub = rospy.Publisher("gnc/traj_target", TrajPoint, queue_size=10)
        self._target_pose_stamped_pub = rospy.Publisher("gnc/traj_target_pose_stamped", PoseStamped, queue_size=10)

        self._arm_servo_pub = rospy.Publisher('vehicle/servos/4/target_position', Float64, queue_size=10)
        self._suction_pub = rospy.Publisher('vehicle/servos/5/target_position', Float64, queue_size=10)
        self._torpedo_pub = rospy.Publisher('vehicle/servos/2/target_position', Float64, queue_size=10)
        self._marker_pub = rospy.Publisher('vehicle/servos/3/target_position', Float64, queue_size=10)

        self._odom_sub = rospy.Subscriber('gnc/odom', OdometryMsg, self._handle_odom)
        while not self.initialized:
            rospy.sleep(0.1)

        rospy.Timer(rospy.Duration.from_sec(0.05), self._pub_target)

    def abort(self):
        self.traj = None

    def set_trajectory(self, traj):
        assert isinstance(traj, Trajectory)
        self.traj = traj
        self.traj.start()

    def get_robot_state(self):
        return self.pose, self.twist

    def get_position(self):
        return np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])

    def get_orientation(self):
        return quat_to_rpy(self.pose.orientation)

    def get_motion_status(self):
        if self.traj is None:
            return TrajectoryStatus.PENDING

        return self.traj.get_status(self.pose)

    def arm(self, armed):
        try:
            self._arm_srv.call(SetBoolRequest(armed))
        except rospy.ServiceException:
            rospy.logwarn_throttle('[Motion Utils] Cannot arm/disarm: Arm server not responding. (This could be due '
                                   'to running in simulation)')

    def goto_pid(self, pos: typing.Tuple[float],
                   heading: float = None,
                   threshold_lin=0.1, threshold_ang=0.1,
                   block: TrajectoryStatus = TrajectoryStatus.FINISHED):
        start_pose, start_twist = self.get_target()
        try:
            if heading is None:
                heading = Rotation.from_quat(tl(start_pose.orientation)).as_euler('ZYX')[0]
            newtraj = HoldPos(pos, heading)
        except PlanningError:
            rospy.logwarn("Trajectory planning failure!")
            return False
            
        self.set_trajectory(newtraj)
        while self.get_motion_status().value < block.value:
            rospy.sleep(0.1)
        return True

    def goto(self, pos: typing.Tuple[float],
                   heading: float = None, 
                   v=.1, a=.1, j=.4,
                   threshold_lin=0.5, threshold_ang=0.5,
                   block: TrajectoryStatus = TrajectoryStatus.FINISHED):
        start_pose, start_twist = self.get_target()
        try:
            newtraj = LinearTrajectory(start_pose, start_twist, [pos], [heading], v=v, a=a, j=j)
        except PlanningError:
            rospy.logwarn("Trajectory planning failure!")
            return False
        self.set_trajectory(newtraj)
        while self.get_motion_status().value < block.value:
            rospy.sleep(0.1)
        return True

    def goto_relative(self, pos: typing.Tuple[float],
                      heading: float = 0,
                      v=.1, a=.1, j=.4,
                      block: TrajectoryStatus = TrajectoryStatus.FINISHED):
        start_pose, start_twist = self.get_target()

        current_heading = Rotation.from_quat(tl(start_pose.orientation)).as_euler('ZYX')[0]
        current_pos = tl(start_pose.position)

        world_pos = current_pos + np.array([
            math.cos(current_heading) * pos[0] - math.sin(current_heading) * pos[1],
            math.sin(current_heading) * pos[0] + math.cos(current_heading) * pos[1],
            0])
        world_pos[2] = pos[2] # Z is NOT relative!!
        world_heading = heading + current_heading

        print(current_pos, world_pos)
        print(current_heading, world_heading)

        try:
            newtraj = LinearTrajectory(start_pose, start_twist, [world_pos], [world_heading], v=v, a=a, j=j, autowind_headings=False)
        except PlanningError:
            rospy.logwarn("Trajectory planning failure!")
            return False
        self.set_trajectory(newtraj)
        while self.get_motion_status().value < block.value:
            rospy.sleep(0.1)

    def reset(self):
        self.traj = None

    def get_target(self) -> typing.Tuple[Pose, Twist]:
        if self.get_motion_status() == TrajectoryStatus.PENDING:
            # Return current position? or none?
            return self.pose, self.twist
        else:
            req = GetTrajectoryRequest()
            req.curr_time = rospy.Time.now()
            req.len = 1
            req.curr_pose = self.pose
            req.curr_twist = self.twist
            req.dt = .1 # i guess
            res: GetTrajectoryResponse = self.traj.get_points(req)
            return res.poses[0], res.twists[0]

    def _pub_target(self, timer_event):
        pose, twist = self.get_target()
        
        if pose is None or twist is None:
            return

        tp = TrajPoint(pose, twist)
        self._target_pub.publish(tp)

        tp_pose_stamped = PoseStamped()
        tp_pose_stamped.header.frame_id = f'{self._tf_namespace}/odom'
        tp_pose_stamped.pose = pose
        self._target_pose_stamped_pub.publish(tp_pose_stamped)

        if self.traj is not None:
            path = self.traj.as_path()
            path.header.frame_id = f'{self._tf_namespace}/odom'

            self._path_pub.publish(path)

    def _handle_get_traj(self, req):
        response = GetTrajectoryResponse()

        if self.traj is None:
            response.success = False
            return response

        response = self.traj.get_points(req)

        return response

    def _handle_odom(self, msg: OdometryMsg):
        self.pose = msg.pose.pose
        self.twist = twist_body_to_world(msg.pose.pose, msg.twist.twist)
        self.initialized = True

    def shoot_torpedo(self, torpedo: int):
        if torpedo not in [0, 1]:
            raise ValueError(f'Invalid torpedo: {torpedo}')

        if torpedo == 0:
            self._torpedo_pub.publish(-50)
            rospy.sleep(1.0)
            self._torpedo_pub.publish(0)
        elif torpedo == 1:
            self._torpedo_pub.publish(50)
            rospy.sleep(1.0)
            self._torpedo_pub.publish(0)

    def drop_marker(self, marker: int):
        if marker not in [0, 1]:
            raise ValueError(f'Invalid marker: {marker}')

        if marker == 0:
            self._marker_pub.publish(-90)
            rospy.sleep(1.0)
            self._marker_pub.publish(0)
        elif marker == 1:
            self._marker_pub.publish(90)
            rospy.sleep(1.0)
            self._marker_pub.publish(0)
