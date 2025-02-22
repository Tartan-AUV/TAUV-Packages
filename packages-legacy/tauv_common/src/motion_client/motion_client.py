import rospy
from threading import Lock, Event
from typing import Optional, Tuple, Any
from spatialmath import SE3, SO3, SE2, Twist3

from tauv_msgs.srv import GetTrajectory, GetTrajectoryRequest, GetTrajectoryResponse
from tauv_msgs.msg import NavigationState
from std_srvs.srv import SetBool
from trajectories import Trajectory, ConstantAccelerationTrajectory, ConstantAccelerationTrajectoryParams
from geometry_msgs.msg import Pose as PoseMsg, PoseStamped as PoseStampedMsg, Twist as TwistMsg, TwistStamped as TwistStampedMsg
from tauv_util.spatialmath import ros_nav_state_to_se3, ros_nav_state_to_body_twist3, body_twist3_to_world_twist3, flatten_se3, flatten_twist3, se3_to_ros_pose, twist3_to_ros_twist

class MotionClient:

    def __init__(self):
        self._tf_namespace: str = None
        self._params: {str: ConstantAccelerationTrajectoryParams} = {}

        self._odom_lock: Lock = Lock()
        self._odom: Optional[Tuple[SE3, Twist3]] = None
        self._odom_target: Optional[Tuple[SE3, Twist3]] = None

        self._trajectory_lock: Lock = Lock()
        self._trajectory: Optional[Trajectory] = None
        self._trajectory_start_time: Optional[rospy.Time] = None
        self._trajectory_complete_event: Event = Event()

        self._load_config()

        self._nav_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_nav_state)

        self._debug_target_pose: rospy.Publisher = rospy.Publisher('gnc/debug_target_pose', PoseStampedMsg)
        self._debug_target_twist: rospy.Publisher = rospy.Publisher('gnc/debug_target_twist', TwistStampedMsg)

        self._get_trajectory_server: rospy.Service = rospy.Service('gnc/get_trajectory', GetTrajectory, self._handle_get_trajectory)

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def _handle_get_trajectory(self, req: GetTrajectoryRequest) -> GetTrajectoryResponse:
        with self._trajectory_lock:
            res = GetTrajectoryResponse()

            if self._trajectory is None:
                if self._odom is None and self._odom_target is None:
                    res.success = False
                    res.message = "no trajectory or odometry"
                    return res

                odom = self._odom_target if self._odom_target is not None else self._odom

                res.poses = [se3_to_ros_pose(odom[0])]
                res.twists = [twist3_to_ros_twist(Twist3())]
                self._publish_debug_target(req.curr_time, res.poses[0], res.twists[0])
                res.success = True
                res.message = "no trajectory, returning current pose"
                return res

            traj_time = (req.curr_time - self._trajectory_start_time).to_sec()
            poses = [None] * req.len
            twists = [None] * req.len

            if traj_time > self._trajectory.duration:
                self._trajectory_complete_event.set()

            for i in range(req.len):
                t = traj_time + i * req.dt
                pose, twist = self._trajectory.evaluate(t)
                poses[i] = se3_to_ros_pose(pose)
                twists[i] = twist3_to_ros_twist(twist)

            res.poses = poses
            res.twists = twists
            self._publish_debug_target(req.curr_time, res.poses[0], res.twists[0])
            res.success = True
            res.message = "success"
            return res

    def arm(self, arm: bool):
        try:
            self._arm_srv(arm)
        except:
            pass

    def goto(self,
             pose: SE3,
             params: Optional[ConstantAccelerationTrajectoryParams] = None,
             flat: bool = True,
             current_time: Optional[rospy.Time] = None):

        params = self._params["default"] if params is None else params
        current_time = rospy.Time.now() if current_time is None else current_time

        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, start_world_twist = start
        if flat:
            start_pose = flatten_se3(start_pose)
            start_world_twist = flatten_twist3(start_world_twist)
            pose = flatten_se3(pose)

        rospy.loginfo(start_pose)
        rospy.loginfo(pose)

        traj = ConstantAccelerationTrajectory(
            start_pose, start_world_twist,
            pose, Twist3(),
            params,
            relax=True,
        )

        self._set_trajectory(traj, current_time)

    def goto_relative(self,
                      relative_pose: SE3,
                      params: Optional[ConstantAccelerationTrajectoryParams] = None,
                      flat: bool = True,
                      current_time: Optional[rospy.Time] = None):

        current_time = rospy.Time.now() if current_time is None else current_time

        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, _ = start
        if flat:
            start_pose = flatten_se3(start_pose)
            relative_pose = flatten_se3(relative_pose)

        pose = start_pose @ relative_pose

        self.goto(pose, params, flat=True, current_time=current_time)

    def goto_relative_with_depth(self,
                                 relative_pose: SE2,
                                 z: float,
                                 params: Optional[ConstantAccelerationTrajectoryParams] = None,
                                 current_time: Optional[rospy.Time] = None):
        current_time = rospy.Time.now() if current_time is None else current_time

        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, _ = start
        start_pose = flatten_se3(start_pose)

        pose = start_pose @ SE3(relative_pose)
        t = pose.t
        t[2] = z
        pose = SE3.Rt(SO3(pose), t)

        self.goto(pose, params, flat=True, current_time=current_time)

    def wait_until_complete(self, timeout: Optional[rospy.Duration] = None) -> bool:
        rospy.logdebug('wait_until_complete')

        with self._trajectory_lock:
            if self._trajectory is None:
                return

        return self._trajectory_complete_event.wait(timeout.to_sec())

    def cancel(self):
        with self._trajectory_lock:
            self._trajectory = None
            self._trajectory_start_time = None
            self._trajectory_complete_event.set()

            self._odom_target = self._odom

    def _set_trajectory(self, trajectory: Trajectory, current_time: rospy.Time):
        with self._trajectory_lock:
            self._trajectory_start_time = current_time
            self._trajectory = trajectory
            self._trajectory_complete_event.clear()

    def _publish_debug_target(self, time: rospy.Time, pose: PoseMsg, twist: TwistMsg):
        pose_stamped = PoseStampedMsg()
        pose_stamped.header.frame_id = f'{self._tf_namespace}/odom'
        pose_stamped.header.stamp = time
        pose_stamped.pose = pose
        self._debug_target_pose.publish(pose_stamped)

        twist_stamped = TwistStampedMsg()
        twist_stamped.header.frame_id = f'{self._tf_namespace}/odom'
        twist_stamped.header.stamp = time
        twist_stamped.twist.linear = twist.linear
        twist_stamped.twist.angular = twist.angular
        self._debug_target_twist.publish(twist_stamped)

    def _get_start(self, time: rospy.Time) -> Optional[Tuple[SE3, Twist3]]:
        with self._trajectory_lock:
            if self._trajectory is not None:
                traj_time = (time - self._trajectory_start_time).to_sec()

                return self._trajectory.evaluate(traj_time)

        with self._odom_lock:
            if self._odom is not None:
                return self._odom

        return None

    def _handle_trajectory_complete(self, timer_event: Any):
        self._trajectory_complete_event.set()

    def _handle_nav_state(self, msg: NavigationState):
        pose = ros_nav_state_to_se3(msg)
        body_twist = ros_nav_state_to_body_twist3(msg)
        world_twist = body_twist3_to_world_twist3(pose, body_twist)
        with self._odom_lock:
            self._odom = (pose, world_twist)

    def get_trajectory_params(self, name: str) -> ConstantAccelerationTrajectoryParams:
        return self._params.get(name)

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')

        params = rospy.get_param('motion/params')
        self._params = {}

        for key, value in params.items():
            self._params[key] = ConstantAccelerationTrajectoryParams(
                v_max_linear=value["v_max_linear"],
                v_max_angular=value["v_max_angular"],
                a_linear=value["a_linear"],
                a_angular=value["a_angular"],
            )
