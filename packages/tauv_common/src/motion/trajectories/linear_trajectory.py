import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseStamped, Twist, Vector3, Quaternion
from nav_msgs.msg import Path
from tauv_msgs.srv import GetTrajectoryRequest, GetTrajectoryResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import linear_distance, yaw_distance, quat_to_rpy, rpy_to_quat, twist_world_to_body, twist_body_to_world

from motion.trajectories.trajectories import Trajectory, TrajectoryStatus
from motion.trajectories.pyscurve import ScurvePlanner


class Waypoint:
    def __init__(self, pose: Pose, linear_error: float = 0.25, angular_error: float = 0.25):
        self.pose: Pose = pose
        self.linear_error: float = linear_error
        self.angular_error: float = angular_error

    # def __init__(self, position: np.array, linear_error: float = 0.25, angular_error: float = 0.25):
    #     self.pose = Pose(tm(position, Vector3), Quaternion(0.0, 0.0, 0.0, 1.0))
    #     self.linear_error: float = linear_error
    #     self.angular_error: float = angular_error

    def to_pose_stamped(self) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.pose = self.pose
        return ps


class LinearTrajectory(Trajectory):
    def __init__(self, waypoints: [Waypoint], linear_constraints: tuple=(1,1,1), angular_constraints: tuple=(1,1,1)):
        self._waypoints: [Waypoint] = waypoints
        self._target: int = 1

        self._linear_constraints: (float, float, float) = linear_constraints
        self._angular_constraints: (float, float, float) = angular_constraints

        self._status = TrajectoryStatus.PENDING

        self._p = ScurvePlanner()

    def _plan(self):
        start = self._waypoints[self._target - 1]
        end = self._waypoints[self._target]

        start_position = tl(start.pose.position) + 1e-4 * np.random.rand(3)
        end_position = tl(end.pose.position)

        length = np.linalg.norm(end_position - start_position)

        self._linear_traj = self._p.plan_trajectory(
            np.array([0.0]), np.array([length]),
            np.array([0.0]), np.array([0.0]),
            v_max=self._linear_constraints[0],
            a_max=self._linear_constraints[1],
            j_max=self._linear_constraints[2],
        )

        start_yaw = quat_to_rpy(start.pose.orientation)[2]
        end_yaw = quat_to_rpy(end.pose.orientation)[2] + 1e-4 * np.random.rand(1)

        # start_body_twist = twist_world_to_body(req.curr_pose, req.curr_twist)
        # start_yaw_velocity = tl(start_body_twist.angular)[2]

        self._yaw_traj = self._p.plan_trajectory(
            np.array([start_yaw]), np.array([end_yaw]),
            np.array([0.0]), np.array([0.0]),
            v_max=self._angular_constraints[0],
            a_max=self._angular_constraints[1],
            j_max=self._angular_constraints[2],
        )

        self._start_time = rospy.Time.now()

        linear_duration = self._linear_traj.time[0]
        yaw_duration = self._yaw_traj.time[0]
        self._segment_duration = rospy.Duration.from_sec(max(linear_duration, yaw_duration))

    def get_points(self, req: GetTrajectoryRequest) -> GetTrajectoryResponse:
        if self._status != TrajectoryStatus.EXECUTING:
            res: GetTrajectoryResponse = GetTrajectoryResponse()
            res.success = False
            return res

        start_waypoint = self._waypoints[self._target - 1]
        target_waypoint = self._waypoints[self._target]

        target_reached = linear_distance(req.curr_pose, target_waypoint.pose) < target_waypoint.linear_error and \
                         yaw_distance(req.curr_pose, target_waypoint.pose) < target_waypoint.angular_error and \
                            np.linalg.norm(tl(req.curr_twist.linear)) < 0.05

        if target_reached and self._target == len(self._waypoints) - 1:
            self._status = TrajectoryStatus.STABILIZED
        elif (req.curr_time - self._start_time).to_sec() > self._segment_duration.to_sec()\
                and self._target == len(self._waypoints) - 1:
            self._status = TrajectoryStatus.FINISHED
        elif target_reached:
            self._target += 1
            self._plan()

        poses = [None] * req.len
        twists = [None] * req.len

        linear_duration = self._linear_traj.time[0]
        yaw_duration = self._yaw_traj.time[0]

        elapsed = (req.curr_time - self._start_time).to_sec()

        for i in range(req.len):
            t = elapsed + req.dt * i
            t_linear = min(t, linear_duration)
            t_yaw = min(t, yaw_duration)

            segment_position = self._linear_traj(t_linear)[0,2]
            segment_linear_velocity = self._linear_traj(t_linear)[0,1]

            start_position = tl(start_waypoint.pose.position)
            target_position = tl(target_waypoint.pose.position)
            segment_direction = target_position - start_position
            normalized_segment_direction = segment_direction / np.linalg.norm(segment_direction)

            position = segment_position * normalized_segment_direction + start_position
            linear_velocity = segment_linear_velocity * normalized_segment_direction

            yaw = self._yaw_traj(t_yaw)[0,2]
            yaw_velocity = self._yaw_traj(t_yaw)[0,1]

            pose = Pose()
            pose.position = tm(position, Vector3)
            pose.orientation = rpy_to_quat(np.array([0.0, 0.0, yaw]))

            twist = Twist()
            twist.linear = tm(linear_velocity, Vector3)
            twist.angular = tm(np.array([0.0, 0.0, yaw_velocity]), Vector3)

            poses[i] = pose
            twists[i] = twist

        res: GetTrajectoryResponse = GetTrajectoryResponse()
        res.poses = poses
        res.twists = twists
        res.auto_twists = False
        res.success = True
        return res

    def get_segment_duration(self) -> rospy.Duration:
        return self._segment_duration

    def start(self):
        self._plan()
        self._status = TrajectoryStatus.EXECUTING

    def get_status(self, pose) -> TrajectoryStatus:
        return self._status

    def as_path(self) -> Path:
        path = Path()
        path.header.frame_id = 'odom_ned'
        path.poses = list(map(lambda w: w.to_pose_stamped(), self._waypoints))
        return path
