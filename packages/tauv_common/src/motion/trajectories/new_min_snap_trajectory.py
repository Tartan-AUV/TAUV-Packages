import rospy
import numpy as np
from math import acos, floor
from scipy.spatial.transform import Rotation
from typing import List, Optional

from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion, Vector3
from nav_msgs.msg import Path

from tauv_msgs.srv import GetTrajectoryRequest, GetTrajectoryResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, rpy_to_quat, twist_world_to_body, twist_body_to_world

from motion.trajectories.pyscurve import ScurvePlanner, Trajectory as ScurveTrajectory
from motion.trajectories.trajectories import Trajectory, TrajectoryStatus
from motion.trajectories.python_optimal_splines.OptimalTrajectory import OptimalTrajectory
from motion.trajectories.python_optimal_splines.OptimalSplineGen import Waypoint, compute_min_derivative_spline
from motion.trajectories.python_optimal_splines.TrajectoryWaypoint import TrajectoryWaypoint

class NewMinSnapTrajectory(Trajectory):

    def __init__(self, start_pose: Pose, start_twist: Twist, poses: [Pose], linear_velocity: float, angular_velocity: float):
        self._orient_forward = True

        # Set error thresholds for each axis
        # When retrieving points, if an axis is exceeding its error threshold
        # AND a correction is not currently running for that axis:
        #  Compute t, the duration of the correction, based on the magnitude of the error
        #  Compute the s-curve correction from the current pose and twist
        #    to the pose and twist of the trajectory evaluated at now + t
        #  Save this trajectory, the start time of the correction, and the finish time of the correction
        #  for the current axis

        # For each axis, if a correction is currently running
        # compute t based on the correction start time and plug the positions and velocities into the returned poses and twists


        self.status = TrajectoryStatus.PENDING
        self._start_pose = start_pose
        self._start_twist = start_twist

        self._correction_planner = ScurvePlanner()
        self._correction_start_times: [Optional[float]] = [None] * 6
        self._correction_end_times: [Optional[float]] = [None] * 6
        self._correction_trajs: [Optional[ScurveTrajectory]] = [None] * 6
        self._correction_thresholds: [float] = [0.01, 0.01, 0.01, 0.02, 0.02, 1e10]
        self._correction_max_velocities: [float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self._correction_max_accelerations: [float] = [1, 1, 1, 1, 1, 1]
        self._correction_duration_scales: [float] = [50, 50, 50, 100, 100, 100]

        position_waypoints: [TrajectoryWaypoint] = []

        start_position = tl(start_pose.position)
        start_orientation = quat_to_rpy(start_pose.orientation)
        start_dirvec = self._quat_to_dirvec(start_pose.orientation)
        start_linear_velocity = tl(start_twist.linear)
        body_start_twist = twist_world_to_body(start_pose, start_twist)
        start_angular_velocity = tl(body_start_twist.angular)

        start_waypoint = TrajectoryWaypoint(tuple(start_position))
        start_waypoint.add_hard_constraints(order=1, values=start_linear_velocity)
        start_waypoint.add_hard_directional_constraint(order=1, values=start_dirvec)
        position_waypoints.append(start_waypoint)

        for (i, pose) in enumerate(poses):
            position = tl(pose.position)
            dirvec = self._quat_to_dirvec(pose.orientation)

            waypoint = TrajectoryWaypoint(tuple(position))

            if i == len(poses) - 1:
                waypoint.add_hard_constraints(order=1, values=[0, 0, 0])

            position_waypoints.append(waypoint)

        self._position_traj = OptimalTrajectory(
            order=5,
            ndims=3,
            waypoints=position_waypoints,
            min_derivative_order=4,
            continuity_order=2,
            constraint_check_dt=0.05
        )

        self._position_traj.solve(
            aggressiveness=None,
            time_opt_order=None,
            use_faster_ts=True,
            T=self._get_duration(start_pose, start_twist, poses, linear_velocity, angular_velocity)
        )

        times = self._position_traj.get_times()

        orientation_waypoints: [TrajectoryWaypoint] = []
        start_orientation_waypoint = TrajectoryWaypoint(tuple(start_orientation))
        start_orientation_waypoint.set_time(0)
        start_orientation_waypoint.add_hard_constraints(order=1, values=start_angular_velocity)
        orientation_waypoints.append(start_orientation_waypoint)

        if self._orient_forward:
            orientation_times = np.arange(0, times[-1], 0.5)

            last_yaw = 0

            for (i, time) in enumerate(orientation_times[1:]):
                linear_velocity = self._position_traj.val(time, dim=None, order=1)
                norm_linear_velocity = linear_velocity / np.linalg.norm(linear_velocity)

                orientation = quat_to_rpy(self._dirvec_to_quat(norm_linear_velocity))
                orientation[0:2] = 0

                yaw = (orientation[2] + np.pi) % (2 * np.pi) - np.pi
                last_yaw = (last_yaw + np.pi) % (2 * np.pi) - np.pi
                if yaw - last_yaw < -np.pi:
                    yaw += 2 * np.pi
                if yaw - last_yaw > np.pi:
                    yaw -= 2 * np.pi

                last_yaw = yaw

                orientation[2] = yaw

                waypoint = TrajectoryWaypoint(tuple(orientation))
                waypoint.set_time(time)

                if i == len(orientation_times) - 1:
                    waypoint.add_hard_constraints(order=1, values=[0, 0, 0])

                orientation_waypoints.append(waypoint)
        else:
            for (i, pose) in enumerate(poses):
                orientation = quat_to_rpy(pose.orientation)

                waypoint = TrajectoryWaypoint(tuple(orientation))
                waypoint.set_time(times[i + 1])

                if i == len(poses) - 1:
                    waypoint.add_hard_constraints(order=1, values=[0, 0, 0])

                orientation_waypoints.append(waypoint)

        self._orientation_traj = OptimalTrajectory(
            order=5,
            ndims=3,
            waypoints=orientation_waypoints,
            min_derivative_order=4,
            continuity_order=2,
            constraint_check_dt=0.05
        )

        self._orientation_traj.solve(
            aggressiveness=None,
            time_opt_order=None,
            use_faster_ts=True,
            T=times[-1],
            skip_times=True,
        )

        self._start_time: rospy.Time = rospy.Time.now()
        self._duration: rospy.Duration = rospy.Duration.from_sec(times[-1])

        self.status = TrajectoryStatus.INITIALIZED

    def _get_duration(self, start_pose: Pose, start_twist: Twist, poses: [Pose], linear_velocity: float, angular_velocity: float) -> float:
        durations: [float] = []

        all_poses = [start_pose] + poses

        for i in range(len(all_poses) - 1):
            start_position = np.array(tl(all_poses[i].position))
            end_position = np.array(tl(all_poses[i + 1].position))
            start_orientation = Rotation.from_quat(tl(all_poses[i].orientation))
            end_orientation = Rotation.from_quat(tl(all_poses[i + 1].orientation))

            position_dist = np.linalg.norm(start_position - end_position)
            orientation_dist = (start_orientation * end_orientation.inv()).as_quat()[3]

            duration = max(position_dist / linear_velocity, orientation_dist / angular_velocity)
            durations.append(duration)

        return sum(durations)

    def get_points(self, req: GetTrajectoryRequest):
        if self.status == TrajectoryStatus.PENDING:
            res: GetTrajectoryResponse = GetTrajectoryResponse()
            res.poses = []
            res.twists = []
            res.auto_twists = False
            res.success = False
            return res

        elapsed = (req.curr_time - self._start_time).to_sec()

        poses: [Pose] = [None] * req.len
        twists: [Pose] = [None] * req.len

        for i in range(req.len):
            t = (req.dt * i) + elapsed

            if t > self._duration.to_sec():
                t = self._duration.to_sec()

            try:
                position = self._position_traj.val(t, dim=None, order=0)
                linear_velocity = self._position_traj.val(t, dim=None, order=1)
                orientation = self._orientation_traj.val(t, dim=None, order=0)
                angular_velocity = self._orientation_traj.val(t, dim=None, order=1)

                pose = Pose()
                pose.position = tm(position, Vector3)
                pose.orientation = rpy_to_quat(orientation)

                body_twist = Twist()
                body_twist.angular = tm(angular_velocity, Vector3)
                world_twist = twist_body_to_world(pose, body_twist)
                world_twist.linear = tm(linear_velocity, Vector3)

                poses[i] = pose
                twists[i] = world_twist
            except Exception as e:
                print('error evaluating trajectory', e)

        for i in range(6):
            if self._correction_end_times[i] is not None \
                and elapsed > self._correction_end_times[i]:
                self._correction_start_times[i] = None
                self._correction_end_times[i] = None
                self._correction_trajs[i] = None

        current_position = tl(req.curr_pose.position)
        current_orientation = quat_to_rpy(req.curr_pose.orientation)
        current_angular_velocity = tl(twist_world_to_body(req.curr_pose, req.curr_twist).angular)
        current_linear_velocity = tl(req.curr_twist.linear)
        expected_position = tl(poses[0].position)
        expected_orientation = quat_to_rpy(poses[0].orientation)

        current_pose = np.concatenate((current_position, current_orientation))
        current_twist = np.concatenate((current_linear_velocity, current_angular_velocity))
        expected_pose = np.concatenate((expected_position, expected_orientation))

        for i in range(6):
            if self._correction_start_times[i] is not None \
                or abs(current_pose[i] - expected_pose[i]) <= self._correction_thresholds[i]:
                continue

            print('correcting', i)
            print(current_pose[i], expected_pose[i])

            correction_duration = self._correction_duration_scales[i] * abs(current_pose[i] - expected_pose[i])

            target_position = self._position_traj.val(elapsed + correction_duration, dim=None, order=0)
            target_linear_velocity = self._position_traj.val(elapsed + correction_duration, dim=None, order=1)
            target_orientation = self._orientation_traj.val(elapsed + correction_duration, dim=None, order=0)
            target_angular_velocity = self._orientation_traj.val(elapsed + correction_duration, dim=None, order=0)

            target_pose = np.concatenate((target_position, target_orientation))
            target_twist = np.concatenate((target_linear_velocity, target_angular_velocity))

            try:
                self._correction_trajs[i] = self._correction_planner.plan_trajectory(
                    [current_pose[i]], [target_pose[i]],
                    [current_twist[i]], [target_twist[i]],
                    v_max=self._correction_max_velocities[i],
                    a_max=self._correction_max_accelerations[i],
                    j_max=1e10,
                    t=correction_duration
                )

                self._correction_start_times[i] = elapsed
                self._correction_end_times[i] = elapsed + correction_duration
            except:
                print('correction failed')

        for i in range(req.len):
            t = (req.dt * i) + elapsed

            if t > self._duration.to_sec():
                t = self._duration.to_sec()

            for j in range(6):
                if self._correction_start_times[j] is not None \
                    and t > self._correction_start_times[j] \
                    and t <= self._correction_end_times[j]:

                    a = self._correction_trajs[j](t - self._correction_start_times[j])[0, 2]
                    da = self._correction_trajs[j](t - self._correction_start_times[j])[0, 1]

                    trajectory_position = tl(poses[i].position)
                    trajectory_linear_velocity = tl(twists[i].linear)
                    trajectory_orientation = quat_to_rpy(poses[i].orientation)
                    trajectory_angular_velocity = tl(twist_world_to_body(poses[i], twists[i]).angular)

                    trajectory_pose = np.concatenate((trajectory_position, trajectory_orientation))
                    trajectory_twist = np.concatenate((trajectory_linear_velocity, trajectory_angular_velocity))

                    trajectory_pose[j] = a
                    trajectory_twist[j] = da

                    poses[i].position = tm(trajectory_pose[0:3], Vector3)
                    poses[i].orientation = rpy_to_quat(trajectory_pose[3:6])

                    body_twist = Twist()
                    body_twist.angular = tm(trajectory_twist[3:6], Vector3)
                    world_twist = twist_body_to_world(poses[i], body_twist)
                    world_twist.linear = tm(trajectory_twist[0:3], Vector3)

                    twists[i] = world_twist

        res: GetTrajectoryResponse = GetTrajectoryResponse()
        res.poses = poses
        res.twists = twists
        res.auto_twists = False
        res.success = True
        return res

    def duration(self) -> rospy.Duration:
        return self._duration

    def time_remaining(self) -> rospy.Duration:
        return (self._start_time + self._duration) - rospy.Time.now()

    def set_executing(self):
        self.status = TrajectoryStatus.EXECUTING

    def get_status(self) -> TrajectoryStatus:
        if self.time_remaining().to_sec() <= 0:
            self.status = TrajectoryStatus.STABILIZED

        return self.status

    def as_path(self, dt=0.1) -> Path:
        if self.status == TrajectoryStatus.PENDING:
            return Path()

        req: GetTrajectoryRequest = GetTrajectoryRequest()
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = 'odom'

        req.curr_time = self._start_time
        req.curr_pose = self._start_pose
        req.curr_twist = self._start_twist
        req.len = floor(self._duration.to_sec() / dt)
        req.dt = dt

        res: GetTrajectoryResponse = self.get_points(req)

        path: Path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()
        path.poses = []

        for (i, pose) in enumerate(res.poses):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'odom'
            pose_stamped.header.stamp = self._start_time + rospy.Duration.from_sec(i * dt)
            pose_stamped.pose = pose
            path.poses.append(pose_stamped)

        return path

    def _quat_to_dirvec(self, orientation: Quaternion) -> List:
        return Rotation.from_quat(tl(orientation)).apply([1, 0, 0])

    def _dirvec_to_quat(self, dirvec: np.array) -> Quaternion:
        quat = Rotation.from_rotvec(acos(np.dot([1, 0, 0], dirvec)) * np.cross([1, 0, 0], dirvec)).as_quat()
        return tm(quat, Quaternion)