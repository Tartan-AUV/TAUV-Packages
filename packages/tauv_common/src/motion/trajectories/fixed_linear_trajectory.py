import rospy
from motion.trajectories.trajectories import TrajectoryStatus, Trajectory

from tauv_msgs.srv import GetTrajectoryRequest, GetTrajectoryResponse
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point, Twist, PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import numpy as np
from collections import Iterable
from tauv_util.types import tl, tm

# imports for s-curve traj lib:
from motion.trajectories.pyscurve import ScurvePlanner

# math
from math import sin, cos, atan2, sqrt, ceil, fabs
import collections

ScurveParams = collections.namedtuple('ScurveParams', 'v_max a_max j_max')


class LinearSegment(object):
    def __init__(self, p0, p1, h0, h1, params_lin, params_ang, start_velocity=(0, 0, 0), start_angular_velocity=0):
        self.p0 = np.array(p0)
        p_vec = np.array(p1) - np.array(p0)
        dist = np.linalg.norm(p_vec)
        self.direction = p_vec / dist # Unit vector pointing along trajectory

        p = ScurvePlanner()

        start_velocity_aligned = np.dot(np.array(start_velocity), self.direction)

        # Build s-curve 1-dof trajectory for position:
        q0 = [0]
        q1 = [dist]
        v0 = [np.sign(start_velocity_aligned)*min(abs(start_velocity_aligned), params_lin.v_max-0.02)]
        v1 = [0]
        v_max = params_lin.v_max
        a_max = params_lin.a_max
        j_max = params_lin.j_max
        self.tr_lin = p.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

        # Build s-curve 1-dof trajectory for heading:
        q0 = [h0]
        q1 = [h1]
        v0 = [np.sign(start_angular_velocity)*min(abs(start_angular_velocity), params_ang.v_max-0.02)]
        v1 = [0]
        v_max = params_ang.v_max
        a_max = params_ang.a_max
        j_max = params_ang.j_max
        self.tr_ang = p.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

    def duration(self):
        return max(self.tr_lin.time[0], self.tr_ang.time[0])

    def __call__(self, t, order=0):
        if t < 0:
            t = 0

        # Get linear positions:
        d = self.tr_lin(min(t, self.tr_lin.time[0]))[0][2-order]
        if order == 0:
            p = list(self.direction * d + self.p0)
        else:
            p = list(self.direction * d)
        h = self.tr_ang(min(t, self.tr_ang.time[0]))[0][2-order]

        res = [p[0], p[1], p[2], h]
        return res


# Uses linear segments between waypoints, with a s-curve velocity
# profile for each waypoint. These trajectories are NOT smooth, and require
# stopping at each point. Initial velocity in the direction of the first segment is
# accounted for, however velocity perpendicular to the first segment is not.
class LinearTrajectory(Trajectory):
    def __init__(self, start_pose, start_twist, positions, headings=None,
                       v=0.4, a=0.4, j=0.4,
                       autowind_headings=True,
                       threshold_pos=0.5, threshold_heading=0.5):
        # MinSnapTrajectory allows a good deal of customization:
        #
        # - start_pose: unstamped Pose TODO: support stamped poses/twists in other frames!
        # - start_twist: unstamped Twist (measured in world frame!)
        # - positions: The position waypoints as a Point[].
        # - headings: heading angles corresponding to positions as a float[].
        #             If None: auto-orient forwards for full trajectory.
        # - v, a, j: max velocity, acceleration, and jerk respectively. (Applies to linear *and* angular trajectories)
        # - autowind_headings: If true, the headings will be automatically winded to reduce the amount of yaw required
        #                      to achieve them. If false, then the robot will use absolute headings which may result in
        #                      yaw motions greater than pi radians. Setting this to false can be useful for forcing the
        #                      robot to do a 360 spin, for example.
        #
        self.status = TrajectoryStatus.PENDING

        assert(len(positions) > 0)
        assert(all([len(e) == 3 for e in positions]))
        if headings is not None:
            if (len(positions) == 1 and type(headings) is float):
                headings = [headings]
            assert(len(headings) == len(positions))

        start_pos = tl(start_pose.position) + 1e-4 * np.random.rand(3)
        start_psi = Rotation.from_quat(tl(start_pose.orientation)).as_euler("ZYX")[0] + 1e-4 * np.random.rand(1)
        start_vel = tl(start_twist.linear)
        start_ang_vel = tl(start_twist.angular)

        # Create list of waypoints (position and heading + velocities for each)
        p = [start_pos] + positions
        pv = [start_vel] + [0] * len(positions)

        if headings is None:
            headings = [None for i in range(len(positions))]
            autowind_headings = True

        h = [start_psi]
        last_psi = start_psi
        for i in range(len(headings)):
            if headings[i] is None:
                dp = np.array(p[i+1]) - np.array(p[i])
                psi = atan2(dp[1], dp[0])
            else:
                psi = headings[i]

            if autowind_headings:
                psi = (psi + np.pi) % (2 * np.pi) - np.pi
                last_psi = (last_psi + np.pi) % (2 * np.pi) - np.pi
                if psi - last_psi < -np.pi:
                    psi += 2 * np.pi
                if psi - last_psi > np.pi:
                    psi -= 2 * np.pi
                last_psi = psi

            h.append(psi)

        hv = [start_ang_vel[2]] + [0] * len(positions)

        self.final_pos = positions[-1]
        self.final_h = h[-1]
        self.threshold_pos = threshold_pos
        self.threshold_h = threshold_heading

        # Create trajectories for each segment:
        lin_params = ScurveParams(v, a, j)
        ang_params = ScurveParams(v, a, j)
        self.segments = []
        self.ts = []
        for i in range(len(p) - 1):
            s = LinearSegment(p[i], p[i+1], h[i], h[i+1], lin_params, ang_params,
                              start_velocity=pv[i], start_angular_velocity=hv[i])
            t = s.duration()
            if i > 0:
                t += self.ts[i-1]
            self.ts.append(t)
            self.segments.append(s)

        self.start_time = rospy.Time.now().to_sec()
        self.status = TrajectoryStatus.PENDING

    def get_points(self, request):
        assert(isinstance(request, GetTrajectoryRequest))

        res = GetTrajectoryResponse()

        poses = []
        twists = []

        elapsed = request.curr_time.to_sec() - self.start_time
        T = self.duration().to_sec()
        for i in range(request.len):
            t = request.dt * i + elapsed
            if t > T:
                t = T

            # Find appropriate segment:
            seg = 0
            while self.ts[seg] <= t:
                seg += 1
                if seg > len(self.ts) - 1:
                    seg = len(self.ts) - 1
                    break

            p = self.segments[seg](t, order=0)
            v = self.segments[seg](t, order=1)

            pose = Pose(tm(p[0:3], Point), tm(Rotation.from_euler('ZYX', [p[3], 0, 0]).as_quat(), Quaternion))
            twist = Twist(tm(v[0:3], Vector3), Vector3(0, 0, v[3]))
            poses.append(pose)
            twists.append(twist)

        res.twists = twists
        res.poses = poses
        res.auto_twists = False
        res.success = True
        return res

    def duration(self):
        return rospy.Duration(self.ts[-1])

    def time_remaining(self):
        end_time = rospy.Time(self.start_time) + self.duration()
        return end_time - rospy.Time.now()

    def start(self):
        self.status = TrajectoryStatus.EXECUTING

    def get_status(self, pose: Pose):
        if self.time_remaining().to_sec() <= 0 and self.status.value < TrajectoryStatus.FINISHED.value:
            self.status = TrajectoryStatus.FINISHED

        R: Rotation = Rotation.from_quat(tl(pose.orientation))
        if self.status == TrajectoryStatus.FINISHED and \
                np.linalg.norm(np.array(self.final_pos) - np.array(tl(pose.position))) <= self.threshold_pos and \
                fabs(self.final_h - R.as_euler('ZYX')[0]) <= self.threshold_h:
            self.status = TrajectoryStatus.STABILIZED

        return self.status

    def as_path(self, dt=0.1):
        request = GetTrajectoryRequest()
        request.curr_pose = Pose()
        request.curr_twist = Twist()
        request.len = int(ceil(self.duration().to_sec()/dt))
        request.dt = dt
        request.curr_time = rospy.Time.from_sec(self.start_time)
        res = self.get_points(request)

        start_time = rospy.Time.now()

        path = Path()
        path.header.frame_id = 'odom_ned'
        path.header.stamp = start_time

        stamped_poses = []
        for i, p in enumerate(res.poses):
            ps = PoseStamped()
            ps.header.stamp = start_time + rospy.Duration.from_sec(dt * i)
            ps.pose = p
            stamped_poses.append(ps)
        path.poses = stamped_poses
        return path
