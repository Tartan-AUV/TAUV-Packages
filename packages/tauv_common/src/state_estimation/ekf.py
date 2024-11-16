from math import sin, cos, tan, pi
from enum import IntEnum
from typing import List, Optional
import time

import rospy
from geometry_msgs.msg import Vector3
import numpy as np


class StateIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    AX = 6
    AY = 7
    AZ = 8
    YAW = 9
    PITCH = 10
    ROLL = 11
    VYAW = 12
    VPITCH = 13
    VROLL = 14


class EKF:
    NUM_FIELDS = 15

    def __init__(self, dvl_offset: np.array, process_covariance: np.array):
        self._dvl_offset: np.array = dvl_offset
        self._process_covariance: np.array = process_covariance

        self._state: np.array = np.zeros(EKF.NUM_FIELDS, float)
        self._covariance: np.array = 1e-9 * np.identity(EKF.NUM_FIELDS, float)

        self._time: Optional[rospy.Time] = None

    def get_state(self, time: rospy.Time) -> (Vector3, Vector3, Vector3, Vector3, Vector3):
        if self._time is None or time <= self._time:
            raise ValueError('get_state for time earlier than last time')

        delta_t: float = time.to_sec() - self._time.to_sec()

        state = self._extrapolate_state(delta_t)

        position = Vector3(state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z])
        velocity = Vector3(state[StateIndex.VX], state[StateIndex.VY], state[StateIndex.VZ])
        acceleration = Vector3(state[StateIndex.AX], state[StateIndex.AY], state[StateIndex.AZ])
        orientation = Vector3(state[StateIndex.ROLL], state[StateIndex.PITCH], state[StateIndex.YAW])
        angular_velocity = Vector3(state[StateIndex.VROLL], state[StateIndex.VPITCH], state[StateIndex.VYAW])
        return position, velocity, acceleration, orientation, angular_velocity

    def handle_imu_measurement(self, orientation: np.array, linear_acceleration: np.array, covariance: np.array, timestamp: rospy.Time):
        self._predict(timestamp)

        fields = [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL,
                  StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        H: np.array = self._get_H(fields)
        z: np.array = np.concatenate((np.flip(orientation), linear_acceleration))
        y: np.array = z - H @ self._state
        y = self._wrap_angles(y, [0, 1, 2])

        self._update(y, covariance, fields)

    def handle_dvl_measurement(self, linear_velocity: np.array, covariance: np.array, timestamp: rospy.Time):
        self._predict(timestamp)

        fields = [StateIndex.VX, StateIndex.VY, StateIndex.VZ]

        H: np.array = self._get_H(fields)
        z: np.array = linear_velocity - self._get_dvl_tangential_velocity()
        y: np.array = z - H @ self._state

        self._update(y, covariance, fields)

    def handle_depth_measurement(self, depth: np.array, covariance: np.array, timestamp: rospy.Time):
        self._predict(timestamp)

        fields = [StateIndex.Z]

        H: np.array = self._get_H(fields)
        z: np.array = depth
        y: np.array = z - H @ self._state

        self._update(y, covariance, fields)

    def _predict(self, timestamp: rospy.Time):
        if self._time is None:
            self._time = timestamp

        delta_t : float = timestamp.to_sec() - self._time.to_sec()
        self._time = timestamp

        self._state = self._extrapolate_state(delta_t)
        self._covariance = self._extrapolate_covariance(delta_t)

    def _update(self, innovation: np.array, covariance: np.array, fields: [int]):
        H: np.array = self._get_H(fields)

        y: np.array = innovation

        R: np.array = np.diag(covariance)

        S: np.array = (H @ self._covariance) @ np.transpose(H) + R

        K: np.array = (self._covariance @ np.transpose(H)) @ np.linalg.pinv(S)

        I: np.array = np.identity(EKF.NUM_FIELDS, float)

        self._state = self._state + np.matmul(K, y)
        self._state = self._wrap_angles(self._state, [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL])

        self._covariance = ((I - (K @ H)) @ self._covariance) @ np.transpose(I - (K @ H))
        self._covariance = self._covariance + (K @ R) @ np.transpose(K)
        self._covariance = np.maximum(np.abs(self._covariance), 1e-9 * np.identity(EKF.NUM_FIELDS, float))

    def _extrapolate_state(self, dt: float) -> np.array:
        F: np.array = self._get_F(dt)

        state = np.matmul(F, self._state)
        state = self._wrap_angles(state, [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL])

        return state

    def _wrap_angles(self, state: np.array, fields: [int]) -> np.array:
        state = state.copy()
        state[fields] = (state[fields] + pi) % (2 * pi) - pi
        return state

    def _extrapolate_covariance(self, dt: float) -> np.array:
        J: np.array = self._get_J(dt)

        Q: np.array = self._process_covariance

        covariance = np.matmul(J, np.matmul(self._covariance, np.transpose(J))) + dt * Q

        return covariance

    def _get_dvl_tangential_velocity(self) -> np.array:
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        vyaw: float = self._state[StateIndex.VYAW]
        vpitch: float = self._state[StateIndex.VPITCH]
        vroll: float = self._state[StateIndex.VROLL]

        w: np.array = np.array([-sp * vyaw + vroll,
                                cp * sr * vyaw + cr * vpitch,
                                cp * cr * vyaw - sr * vpitch])

        return np.cross(w, self._dvl_offset)

    def _get_H(self, fields: [int]) -> np.array:
        H: np.array = np.zeros((len(fields), EKF.NUM_FIELDS), float)

        for i, f in enumerate(fields):
            H[i, f] = 1

        return H

    def _get_F(self, dt: float) -> np.array:
        cy: float = cos(self._state[StateIndex.YAW])
        sy: float = sin(self._state[StateIndex.YAW])
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        tp: float = tan(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        F: np.array = np.identity(EKF.NUM_FIELDS, float)
        F[StateIndex.X, StateIndex.VX] = (cp * cy) * dt
        F[StateIndex.X, StateIndex.VY] = (cy * sp * sr - cr * sy) * dt
        F[StateIndex.X, StateIndex.VZ] = (cr * cy * sp + sr * sy) * dt
        F[StateIndex.X, StateIndex.AX] = 0.5 * F[StateIndex.X, StateIndex.VX] * dt
        F[StateIndex.X, StateIndex.AY] = 0.5 * F[StateIndex.X, StateIndex.VY] * dt
        F[StateIndex.X, StateIndex.AZ] = 0.5 * F[StateIndex.X, StateIndex.VZ] * dt
        F[StateIndex.Y, StateIndex.VX] = (cp * sy) * dt
        F[StateIndex.Y, StateIndex.VY] = (cp * cy + sp * sr * sy) * dt
        F[StateIndex.Y, StateIndex.VZ] = (-cy * sr + cr * sp * sy) * dt
        F[StateIndex.Y, StateIndex.AX] = 0.5 * F[StateIndex.Y, StateIndex.VX] * dt
        F[StateIndex.Y, StateIndex.AY] = 0.5 * F[StateIndex.Y, StateIndex.VY] * dt
        F[StateIndex.Y, StateIndex.AZ] = 0.5 * F[StateIndex.Y, StateIndex.VZ] * dt
        F[StateIndex.Z, StateIndex.VX] = (-sp) * dt
        F[StateIndex.Z, StateIndex.VY] = (cp * sr) * dt
        F[StateIndex.Z, StateIndex.VZ] = (cp * cr) * dt
        F[StateIndex.Z, StateIndex.AX] = 0.5 * F[StateIndex.Z, StateIndex.VX] * dt
        F[StateIndex.Z, StateIndex.AY] = 0.5 * F[StateIndex.Z, StateIndex.VY] * dt
        F[StateIndex.Z, StateIndex.AZ] = 0.5 * F[StateIndex.Z, StateIndex.VZ] * dt
        F[StateIndex.VX, StateIndex.AX] = dt
        F[StateIndex.VY, StateIndex.AY] = dt
        F[StateIndex.VZ, StateIndex.AZ] = dt
        F[StateIndex.YAW, StateIndex.VYAW] = (cr / cp) * dt
        F[StateIndex.YAW, StateIndex.VPITCH] = (sr / cp) * dt
        F[StateIndex.PITCH, StateIndex.VYAW] = (-sr) * dt
        F[StateIndex.PITCH, StateIndex.VPITCH] = (cr) * dt
        F[StateIndex.ROLL, StateIndex.VYAW] = (cr * tp) * dt
        F[StateIndex.ROLL, StateIndex.VPITCH] = (sr * tp) * dt
        F[StateIndex.ROLL, StateIndex.VROLL] = dt

        return F

    def _get_J(self, dt: float) -> np.array:
        J: np.array = self._get_F(dt)

        cy: float = cos(self._state[StateIndex.YAW])
        sy: float = sin(self._state[StateIndex.YAW])
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        tp: float = tan(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        J[StateIndex.X, StateIndex.YAW] = self._get_partial(-cp * sy,
                                                            -cy * cr - sp * sy * sr,
                                                            cy * sr - sp * sy * cr,
                                                            dt)
        J[StateIndex.X, StateIndex.PITCH] = self._get_partial(-sp * cy,
                                                              cp * cy * sr,
                                                              cp * cy * cr,
                                                              dt)
        J[StateIndex.X, StateIndex.ROLL] = self._get_partial(0,
                                                             sp * cy * cr + sy * sr,
                                                             sy * cr - sp * cy * sr,
                                                             dt)
        J[StateIndex.Y, StateIndex.YAW] = self._get_partial(cp * cy,
                                                            sp * cy * sr - sy * cr,
                                                            sp * cy * sr + sy * sr,
                                                            dt)
        J[StateIndex.Y, StateIndex.PITCH] = self._get_partial(-sp * sy,
                                                              cp * sy * sr,
                                                              cp * sy * cr,
                                                              dt)
        J[StateIndex.Y, StateIndex.ROLL] = self._get_partial(0,
                                                             sp * sy * cr - cy * sr,
                                                             -cy * cr - sp * sy * sr,
                                                             dt)
        J[StateIndex.Z, StateIndex.PITCH] = self._get_partial(-cp,
                                                              -sp * sr,
                                                              -sp * cr,
                                                              dt)
        J[StateIndex.Z, StateIndex.ROLL] = self._get_partial(0,
                                                             cp * sr,
                                                             -cp * sr,
                                                             dt)

        vyaw: float = self._state[StateIndex.VYAW]
        vpitch: float = self._state[StateIndex.VPITCH]

        J[StateIndex.YAW, StateIndex.PITCH] = (tp * sr / cp) * vpitch * dt + (tp * cr / cp) * vyaw * dt
        J[StateIndex.YAW, StateIndex.ROLL] = (cr / cp) * vpitch * dt + (-sr / cp) * vyaw * dt
        J[StateIndex.PITCH, StateIndex.ROLL] = (-sr) * vpitch * dt + (-cr) * vyaw * dt
        J[StateIndex.ROLL, StateIndex.PITCH] = (sr / (cp * cp)) * vpitch * dt + (cr / (cp * cp)) * vyaw * dt
        J[StateIndex.ROLL, StateIndex.ROLL] = 1 + (tp * cr) * vpitch * dt + (-tp * sr) * vyaw * dt

        return J

    def _get_partial(self, xc: float, yc: float, zc: float, dt: float) -> float:
        vx: float = self._state[StateIndex.VX]
        vy: float = self._state[StateIndex.VY]
        vz: float = self._state[StateIndex.VZ]
        ax: float = self._state[StateIndex.AX]
        ay: float = self._state[StateIndex.AY]
        az: float = self._state[StateIndex.AZ]

        return (xc * vx + yc * vy + zc * vz) * dt + \
               (xc * ax + yc * ay + zc * az) * 0.5 * dt * dt