import numpy as np
from math import cos, sin
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion
from scipy.spatial.transform import Rotation
import tf2_ros as tf2

from tauv_util.types import tl, tm

def quat_to_rpy(orientation: Quaternion) -> np.array:
    return np.flip(Rotation.from_quat(tl(orientation)).as_euler('ZYX'))

def rpy_to_quat(orientation: np.array) -> Quaternion:
    return tm(Rotation.from_euler('ZYX', np.flip(orientation)).as_quat(), Quaternion)

def build_pose(position: np.array, orientation: np.array) -> Pose:
    return Pose(tm(position, Vector3), rpy_to_quat(orientation))

def build_twist(linear_velocity: np.array, angular_velocity: np.array) -> Twist:
    return Twist(tm(linear_velocity, Vector3), tm(angular_velocity, Vector3))

def multiply_quat(q1: np.array, q2: np.array) -> np.array:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])


def linear_body_to_world_matrix(pose: Pose) -> np.array:
    orientation = quat_to_rpy(pose.orientation)

    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])
    cy = cos(orientation[2])
    sy = sin(orientation[2])

    body_to_world = np.array([
        [cp * cy, sp * cy * sr - sy * cr, sp * cy * cr + sy * sr],
        [cp * sy, sp * sy * sr + cy * cr, sp * sy * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    return body_to_world


def angular_body_to_world_matrix(pose: Pose) -> np.array:
    orientation = quat_to_rpy(pose.orientation)

    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])

    body_to_world = np.array([
        [1, 0, -sp],
        [0, cr, sr * cp],
        [0, -sr, cr * cp]
    ])

    return body_to_world


def twist_body_to_world(pose: Pose, twist: Twist) -> Twist:
    body_velocity = np.array(tl(twist.linear))
    body_angular_velocity = np.array(tl(twist.angular))

    world_velocity = linear_body_to_world_matrix(pose) @ body_velocity

    world_angular_velocity = angular_body_to_world_matrix(pose) @ body_angular_velocity

    world_twist = Twist(
        linear=tm(world_velocity, Vector3),
        angular=tm(world_angular_velocity, Vector3)
    )

    return world_twist


def twist_world_to_body(pose: Pose, twist: Twist) -> Twist:
    world_velocity = np.array(tl(twist.linear))
    world_angular_velocity = np.array(tl(twist.angular))

    body_velocity = np.linalg.inv(linear_body_to_world_matrix(pose)) @ world_velocity
    body_angular_velocity = np.linalg.inv(angular_body_to_world_matrix(pose)) @ world_angular_velocity

    body_twist = Twist(
        linear=tm(body_velocity, Vector3),
        angular=tm(body_angular_velocity, Vector3)
    )

    return body_twist

def linear_distance(a: Pose, b: Pose) -> float:
    return np.linalg.norm(tl(a.position) - tl(b.position))

def yaw_distance(a: Pose, b: Pose) -> float:
    return np.abs(quat_to_rpy(a.orientation)[2] - quat_to_rpy(b.orientation)[2])

def tf2_transform_to_translation(t: tf2.TransformStamped) -> np.array:
    translation = np.array([
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z
    ])
    return translation

def tf2_transform_to_quat(t: tf2.TransformStamped) -> np.array:
    quat = np.array([
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w
    ])
    return quat

def tf2_transform_to_homogeneous(t: tf2.TransformStamped) -> np.array:
    trans = tf2_transform_to_translation(t)
    quat = tf2_transform_to_quat(t)
    rotm = quat_to_rotm(quat)

    return np.vstack((
        np.hstack((rotm, trans[:, np.newaxis])),
        np.array([0, 0, 0, 1])
    ))

# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
def quat_to_rotm(q: np.array) -> np.array:
    # q is a unit quaternion
    # q = [qi, qj, qk, qr]
    rotm = np.zeros((3, 3))
    rotm[0, 0] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    rotm[0, 1] = 2 * (q[0] * q[1] - q[2] * q[3])
    rotm[0, 2] = 2 * (q[0] * q[2] + q[1] * q[3])
    rotm[1, 0] = 2 * (q[0] * q[1] + q[2] * q[3])
    rotm[1, 1] = 1 - 2 * (q[0] ** 2 + q[2] ** 2)
    rotm[1, 2] = 2 * (q[1] * q[2] - q[0] * q[3])
    rotm[2, 0] = 2 * (q[0] * q[2] - q[1] * q[3])
    rotm[2, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
    rotm[2, 2] = 1 - 2 * (q[0] ** 2 + q[1] ** 2)
    return rotm

def euler_velocity_to_axis_velocity(orientation: np.array, euler_velocity: np.array) -> np.array:
    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])

    T = np.array([
        [1, 0, -sp],
        [0, cr, cp * sr],
        [0, -sr, cp * cr]
    ])

    axis_velocity = T @ euler_velocity

    return axis_velocity

# TODO: is this used?
def axis_velocity_to_euler_velocity(axis_velocity: np.array) -> np.array:
    return axis_velocity

def euler_acceleration_to_axis_acceleration(orientation: np.array, euler_velocity: np.array, euler_acceleration: np.array) -> np.array:
    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])
    dr = euler_velocity[0]
    dp = euler_velocity[1]
    dy = euler_velocity[2]

    T1 = np.array([
        -cp * dp * dy,
        -sr * dp * dr - sp * sr * dp * dy + cp * cr * dr * dy,
        -cr * dp * dr - cr * sp * dp * dy - cp * sr * dr * dy
    ])
    T2 = np.array([
        [1, 0, -sp],
        [0, cr, cp * sr],
        [0, -sr, cp * cr]
    ])

    axis_acceleration = T1 + T2 @ euler_acceleration

    return axis_acceleration


# TODO: Is this used?
def axis_acceleration_to_euler_acceleration(axis_acceleration: np.array) -> np.array:
    return axis_acceleration