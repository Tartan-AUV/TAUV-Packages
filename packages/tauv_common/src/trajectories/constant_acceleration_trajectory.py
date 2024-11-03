import numpy as np
from dataclasses import dataclass
from spatialmath import SE3, SO3, Twist3
from math import sqrt
from typing import Optional

import numpy as np

from trajectories.trajectory import Trajectory


class ConstantAccelerationTrajectory1D:

    def __init__(self,
                 q0: float, v0: float,
                 q1: float, v1: float,
                 v_max: float,
                 a: float,
                 relax: bool = True,
                 t: Optional[float] = None):

        q0 = float(q0)
        v0 = float(v0)
        q1 = float(q1)
        v1 = float(v1)
        v_max = float(v_max)
        a = float(a)
        t = float(t) if t is not None else None

        self._q0 = q0
        self._q1 = q1
        self._v0 = v0
        self._v1 = v1
        self._sign = 1
        self._v = 0
        self._a = 0
        self._t = t if t is not None else 0
        self._ta = 0
        self._td = 0

        # TODO: Allow relaxation of v_max to maximum of start velocity and end velocity
        # TODO: Handle case where endpoints are the same, velocities are different

        if v_max <= 0:
            raise ValueError("v_max must be positive")
        if a <= 0:
            raise ValueError("a must be positive")

        if np.isclose(q0, q1):
            if not np.isclose(v0, v1):
                raise ValueError("requires infinite acceleration")
            else:
                return

        h = q1 - q0
        self._sign = 1
        if q1 < q0:
            self._sign = -1
            q0, q1, v0, v1 = -q0, -q1, -v0, -v1
            h = -h

        if abs(v0) > v_max or abs(v1) > v_max:
            if relax:
                v_max = max(abs(v0), abs(v1))
            else:
                raise ValueError("v0 and v1 cannot exceed v_max")

        v_max = self._sign * v_max

        self._q0 = q0
        self._q1 = q1
        self._v0 = v0
        self._v1 = v1

        a_required = abs((v0 ** 2) - (v1 ** 2)) / (2 * h)
        if t is not None:
            a_limit = (
                (2 * h) - t * (v0 + v1)
                + sqrt(
                      (4 * h ** 2)
                      - (4 * h * (v0 + v1)) * t
                      + 2 * ((v0 ** 2) + (v1 ** 2)) * (t ** 2)
                )
            ) / (t ** 2)
            a_required = max(a_required, a_limit)

        feasible = a >= a_required

        if not feasible:
            if not relax:
                raise ValueError("a is too small")
            else:
                a = a_required

        a = self._sign * a
        self._a = a

        reach_v_max = (h * a) > (v_max ** 2) - (((v0 ** 2) + (v1 ** 2)) / 2)

        if t is not None:
            v = (1 / 2) * (
                v0 + v1 + a * t
                - sqrt(
                (a ** 2) * (t ** 2)
                - 4 * a * h
                + 2 * a * (v0 + v1) * t
                - (v0 - v1) ** 2
                )
            )
        elif reach_v_max:
            v = v_max
        else:
            v = sqrt((h * a) + (((v0 ** 2) + (v1 ** 2)) / 2))

        self._v = v

        self._ta = abs(v - v0) / a
        self._td = abs(v - v1) / a

        if t is not None:
            self._t = t
        elif reach_v_max:
            self._t = \
                (h / v_max) \
                + (v_max / (2 * a)) * ((1 - (v0 / v_max)) ** 2) \
                + (v_max / (2 * a)) * ((1 - (v1 / v_max)) ** 2)
        else:
            self._t = self._ta + self._td

    def evaluate(self, t: float) -> (float, float):
        if t < 0:
            q = self._q0
            v = self._v0
        elif t < self._ta:
            q = self._q0 + self._v0 * t + ((self._v - self._v0) / (2 * self._ta)) * (t ** 2)
            v = self._v0 + ((self._v - self._v0) / self._ta) * t
        elif t < self._t - self._td:
            q = self._q0 + self._v0 * (self._ta / 2) + self._v * (t - (self._ta / 2))
            v = self._v
        elif t < self._t:
            q = self._q1 - self._v1 * (self._t - t) - ((self._v - self._v1) / (2 * self._td)) * ((self._t - t) ** 2)
            v = self._v1 + ((self._v - self._v1) / self._td) * (self._t - t)
        else:
            q = self._q1
            v = self._v1

        return self._sign * q, self._sign * v

    @property
    def duration(self) -> float:
        return self._t


@dataclass
class ConstantAccelerationTrajectoryParams:
    v_max_linear: float
    a_linear: float
    v_max_angular: float
    a_angular: float


class ConstantAccelerationTrajectory(Trajectory):
    # Compute trajectory f: [0, T] -> R for position and trajectory g[0, T] -> R for orienation
    # f outputs distance in m
    # g outputs angle in rad
    # Translation components of poses are start_pose + f(t) * (end_pose - start_pose) / ||end_pose - start_pose||
    # Rotation components of poses are start_pose + slerp(start_pose, end_pose, g(t) / angle(start_pose, end_pose))

    # Allow constraints on start and end velocity parallel to the path
    # Allow constraints on start and end angular velocity parallel to the path

    # Allow constraints on maximum velocity f'(t) and acceleration f''(t)
    # Allow constraints on maximum angular velocity g'(t) and acceleration g''(t)

    # In the event that the trajectory is not feasible (magnitude of start / end twists is too high)...
    # Relax acceleration constraints to make trajectory feasible, return what the corrected acceleration is
    # Only do this if relax_a_max is set

    # Build just enough of an internal representation for fast on-line evaluation of poses and twists at arbitrary times

    # Eventually, add an auto-orient feature that ignores start and end poses, sets poses parallel to the path

    def __init__(self,
                 start_pose: SE3, start_twist: Twist3,
                 end_pose: SE3, end_twist: Twist3,
                 params: ConstantAccelerationTrajectoryParams,
                 relax: bool = True):
        self._start_pose = start_pose
        self._start_twist = start_twist
        self._end_pose = end_pose
        self._end_twist = end_twist

        self._linear_direction = end_pose.t - start_pose.t
        linear_distance = np.linalg.norm(self._linear_direction)
        if not np.isclose(linear_distance, 0):
            self._linear_direction = self._linear_direction / linear_distance

        self._angular_distance = start_pose.angdist(end_pose)
        relative_rotation = SO3(start_pose).inv() * SO3(end_pose)

        linear_start_velocity = np.dot(start_twist.v, self._linear_direction)
        linear_end_velocity = np.dot(end_twist.v, self._linear_direction)

        _, self._angular_direction = relative_rotation.angvec()
        angular_start_velocity = np.dot(start_twist.w, self._angular_direction)
        angular_end_velocity = np.dot(end_twist.w, self._angular_direction)

        linear_traj = ConstantAccelerationTrajectory1D(
            0, linear_start_velocity,
            linear_distance, linear_end_velocity,
            params.v_max_linear, params.a_linear, relax,
        )

        angular_traj = ConstantAccelerationTrajectory1D(
            0, angular_start_velocity,
            self._angular_distance, angular_end_velocity,
            params.v_max_angular, params.a_angular, relax
        )

        if linear_traj.duration > angular_traj.duration:
            angular_traj = ConstantAccelerationTrajectory1D(
                0, angular_start_velocity,
                self._angular_distance, angular_end_velocity,
                params.v_max_angular, params.a_angular, relax,
                t=linear_traj.duration,
            )
        else:
            linear_traj = ConstantAccelerationTrajectory1D(
                0, linear_start_velocity,
                linear_distance, linear_end_velocity,
                params.v_max_linear, params.a_linear, relax,
                t=angular_traj.duration,
            )

        self._linear_traj = linear_traj
        self._angular_traj = angular_traj

    def evaluate(self, time: float) -> (SE3, Twist3):
        linear_q, linear_v = self._linear_traj.evaluate(time)
        angular_q, angular_v = self._angular_traj.evaluate(time)

        position = self._start_pose.t + linear_q * self._linear_direction
        linear_velocity = linear_v * self._linear_direction

        orientation = SO3(self._start_pose) * SO3.AngleAxis(angular_q, self._angular_direction)
        angular_velocity = angular_v * self._angular_direction

        pose = SE3.Rt(orientation, position)
        twist = Twist3(linear_velocity, angular_velocity)

        return pose, twist

    @property
    def duration(self) -> float:
        return max(self._linear_traj.duration, self._angular_traj.duration)
