import numpy as np
from spatialmath.base.types import R3
from spatialmath import SO3, SE3, SE2, Twist3, Twist2, UnitQuaternion
from math import cos, sin

from geometry_msgs.msg import Vector3 as Vector3Msg, Point as PointMsg, Quaternion as QuaternionMsg, Transform as TransformMsg, Pose as PoseMsg, Twist as TwistMsg
from tauv_msgs.msg import NavigationState as NavigationStateMsg


def r3_to_ros_vector3(x: R3) -> Vector3Msg:
    return Vector3Msg(x[0], x[1], x[2])


def ros_vector3_to_r3(x: Vector3Msg) -> R3:
    return np.array([x.x, x.y, x.z])


def r3_to_ros_point(x: R3) -> PointMsg:
    return PointMsg(x[0], x[1], x[2])


def ros_point_to_r3(x: PointMsg) -> R3:
    return np.array([x.x, x.y, x.z])


def ros_quaternion_to_unit_quaternion(x: QuaternionMsg) -> UnitQuaternion:
    return UnitQuaternion(s=x.w, v=np.array([x.x, x.y, x.z]))


def unit_quaternion_to_ros_quaternion(x: UnitQuaternion) -> QuaternionMsg:
    return QuaternionMsg(w=x.s, x=x.v[0], y=x.v[1], z=x.v[2])


def ros_transform_to_se3(x: TransformMsg) -> SE3:
    return SE3.Rt(ros_quaternion_to_unit_quaternion(x.rotation).SO3(), ros_vector3_to_r3(x.translation))


def se3_to_ros_transform(x: SE3) -> TransformMsg:
    return TransformMsg(
        translation=r3_to_ros_vector3(x.t),
        rotation=unit_quaternion_to_ros_quaternion(UnitQuaternion(x)),
    )


def se3_to_ros_pose(x: SE3) -> PoseMsg:
    return PoseMsg(
        r3_to_ros_vector3(x.t),
        unit_quaternion_to_ros_quaternion(UnitQuaternion(x)),
    )


def twist3_to_ros_twist(x: Twist3) -> TwistMsg:
    return TwistMsg(
        r3_to_ros_vector3(x.v),
        r3_to_ros_vector3(x.w)
    )


def ros_nav_state_to_se3(x: NavigationStateMsg) -> SE3:
    orientation = SO3.RPY(ros_vector3_to_r3(x.orientation), order='zyx')
    position = ros_vector3_to_r3(x.position)
    pose = SE3.Rt(orientation, position)
    return pose


def ros_nav_state_to_body_twist3(x: NavigationStateMsg) -> Twist3:
    orientation = SO3.RPY(ros_vector3_to_r3(x.orientation), order='zyx')
    angular_twist = euler_velocity_to_body_twist3(orientation, ros_vector3_to_r3(x.euler_velocity))
    linear_twist = Twist3(ros_vector3_to_r3(x.linear_velocity), np.zeros(3))
    twist = linear_twist + angular_twist
    return twist


def euler_velocity_to_body_twist3(pose: SO3, euler_velocity: R3) -> Twist3:
    # Reference: https://aviation.stackexchange.com/a/84008from
    r, p, y = pose.rpy(order='zyx')
    cr = cos(r)
    sr = sin(r)
    cp = cos(p)
    sp = sin(p)

    T = np.array([
        [1, 0, -sp],
        [0, cr, cp * sr],
        [0, -sr, cp * cr]
    ])

    body_velocity = T @ euler_velocity

    return Twist3(np.zeros(3), body_velocity)


def world_twist3_to_body_twist3(pose: SO3, twist: Twist3) -> Twist3:
    body_twist = Twist3(pose.R @ twist.v, pose.R @ twist.w)
    return body_twist


def body_twist3_to_world_twist3(pose: SO3, twist: Twist3) -> Twist3:
    pose_inv = pose.inv()
    world_twist = Twist3(pose_inv.R @ twist.v, pose_inv.R @ twist.w)
    return world_twist


def flatten_se3(pose: SE3) -> SE3:
    t = pose.t
    yaw = pose.rpy(order='zyx')[2]
    flat_pose = SE3.Rt(SO3.RPY((0, 0, yaw), order='zyx'), t)

    return flat_pose


def flatten_twist3(twist: Twist3) -> Twist3:
    v = twist.v
    w = twist.w
    w[0:2] = 0
    flat_twist = Twist3(v, w)

    return flat_twist

