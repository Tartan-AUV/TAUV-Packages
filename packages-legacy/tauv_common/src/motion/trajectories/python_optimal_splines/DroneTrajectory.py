from motion.trajectories.python_optimal_splines.OptimalTrajectory import OptimalTrajectory
from motion.trajectories.python_optimal_splines.TrajectoryWaypoint import TrajectoryWaypoint
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import Header
import rospy
from math import atan2, asin, sqrt
from scipy import linalg


class DroneGate:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class DroneTrajectory:
    def __init__(self):
        self.gates = []
        self.waypoints = []
        self.splines = []
        self.start_pos = None
        self.start_velocity = None
        self.end_pos = None
        self.end_velocity = None
        self.trajectory = None
        self.direction_radius = 0.1

        self.spacing = 0.2
        self.ext_radius = 0.15  # ensure that this is less than spacing/sqrt(3) to properly handle diagonal gates.
        self.int_radius = 0

    def set_start(self, position, velocity):
        if isinstance(position, np.ndarray):
            position = position.ravel()
        if isinstance(velocity, np.ndarray):
            velocity = velocity.ravel()
        self.start_pos = list(position)
        self.start_velocity = velocity

    def set_end(self, position, velocity):
        if isinstance(position, np.ndarray):
            position = position.ravel()
        if isinstance(velocity, np.ndarray):
            velocity = velocity.ravel()
        self.end_pos = position
        self.end_velocity = velocity

    def clear_gates(self):
        self.gates = []

    def add_gate(self, position, orientation):
        if isinstance(position, np.ndarray):
            position = position.ravel()
        self.gates.append(DroneGate(position, orientation))

    def solve(self, aggressiveness, T=None):
        if self.start_pos is None:
            return None
        if len(self.gates) == 0 and self.end_pos is None:
            return None

        self.waypoints = []

        gate_waypoints = []
        # outer guiding waypoints have lower constraint, full equality constraint at center
        radii = [self.ext_radius, self.int_radius, 0, self.int_radius, self.ext_radius]
        guide_spacing = self.spacing * np.arange(start=-1, stop=1 + 1)
        for gate in self.gates:
            rotm = Rotation.from_quat(gate.orientation).as_dcm()
            dirvec = rotm.dot(np.array([1, 0, 0]))
            for ri, offset in enumerate(guide_spacing):
                radius = radii[ri]
                guide_pos = rotm.dot(np.array([offset, 0, 0])) + np.array(gate.position).transpose()
                if np.isclose(offset, 0, atol=1e-9):
                    # if is true gate, don't allow soft constraint
                    guide_wp = TrajectoryWaypoint(tuple(guide_pos.ravel()))
                    guide_wp.add_soft_directional_constraint(1, tuple(dirvec), self.direction_radius)
                    gate_waypoints.append(guide_wp)
                else:
                    guide_wp = TrajectoryWaypoint(tuple(guide_pos.ravel()))
                    # guide_wp.add_soft_directional_constraint(1, tuple(dirvec), self.direction_radius)
                    # gate_waypoints.append(guide_wp)

        start_waypoint = TrajectoryWaypoint(tuple(self.start_pos))
        start_waypoint.add_hard_constraints(1, tuple(self.start_velocity))

        self.waypoints.append(start_waypoint)
        self.waypoints.extend(gate_waypoints)

        if self.end_pos is not None:
            end_waypoint = TrajectoryWaypoint(tuple(self.end_pos))
            if self.end_velocity is not None:
                end_waypoint.add_hard_constraints(1, tuple(self.end_velocity))
            self.waypoints.append(end_waypoint)

        self.trajectory = OptimalTrajectory(5, 3, self.waypoints)
        self.trajectory.solve(aggressiveness, T=T)

    def val(self, t, order=0, dim=None):
        if self.trajectory is None:
            return None
        last_time = self.trajectory.end_time()
        if t > last_time:
            t = last_time
        return self.trajectory.val(t, dim, order)

    def full_pose(self, time_elapsed):
        pos = self.val(time_elapsed)
        vel = self.val(time_elapsed, order=1)
        unit_vec = np.array(vel) / np.linalg.norm(np.array(vel))

        psi = atan2(unit_vec[1], unit_vec[0])
        theta = asin(-unit_vec[2])
        q = Rotation.from_euler('ZYX', [psi, theta, 0]).as_quat()

        return pos, vel, q

    def as_path(self, dt, start_time, frame='odom'):
        if self.trajectory is None:
            return None
        ts = np.arange(0, self.trajectory.end_time(), dt)

        poses = []
        d = 0
        last_pos = self.val(0)
        for t in ts:
            pos = self.val(t)
            vel = self.val(t, order=1)

            d = d + sqrt(linalg.norm(np.array(pos) - np.array(last_pos)))
            last_pos = pos

            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = start_time + rospy.Duration(t)
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]

            vel = np.array(vel) / np.linalg.norm(np.array(vel))

            psi = atan2(vel[1], vel[0])
            theta = asin(-vel[2])
            q = Rotation.from_euler('ZYX', [psi, theta, 0]).as_quat()
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            poses.append(pose)

        path = Path()
        path.header.frame_id = frame
        path.header.stamp = start_time
        path.poses = poses

        print("Total path length: {}m, time: {}s, average speed: {} m/s".format(d, self.trajectory.end_time(), d / self.trajectory.end_time()))

        return path
