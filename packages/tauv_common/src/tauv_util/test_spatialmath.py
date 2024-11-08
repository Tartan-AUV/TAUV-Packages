import unittest
from spatialmath import SE3, SO3, Twist3, SE2, Twist2
import numpy as np
from math import pi

from tauv_msgs.msg import NavigationState

from tauv_util.spatialmath import \
    ros_nav_state_to_se3, ros_nav_state_to_body_twist3, \
    euler_velocity_to_body_twist3,\
    body_twist3_to_world_twist3, world_twist3_to_body_twist3,\
    flatten_se3, flatten_twist3


class TestEulerVelocityToBodyTwist3(unittest.TestCase):

    def test_identity_pose(self):
        euler_velocity = np.random.rand(3)
        pose = SO3()

        body_twist = euler_velocity_to_body_twist3(pose, euler_velocity)

        self.assertTrue(np.allclose(body_twist.v, np.zeros(3)))
        self.assertTrue(np.allclose(euler_velocity, body_twist.w))

    def test_gimbal_lock(self):
        euler_velocity = np.array([-1.0, 0.0, 1.0])
        pose = SO3.Ry(pi / 2)

        body_twist = euler_velocity_to_body_twist3(pose, euler_velocity)

        self.assertTrue(np.allclose(body_twist.v, np.zeros(3)))
        self.assertTrue(np.allclose(body_twist.w, np.array([-2.0, 0.0, 0.0])))


class TestBodyTwist3ToWorldTwist3(unittest.TestCase):

    def test_identity_pose(self):
        body_twist = Twist3.Rand()
        pose = SO3()

        world_twist = body_twist3_to_world_twist3(pose, body_twist)

        self.assertTrue(np.allclose(body_twist.S, world_twist.S))

    def test_yaw_orientation_linear_velocity(self):
        body_twist = Twist3.Tx(1.0)
        pose = SO3.Rz(pi / 2)

        world_twist = body_twist3_to_world_twist3(pose, body_twist)

        self.assertTrue(np.allclose(world_twist.S, Twist3.Ty(-1.0).S))

    def test_yaw_orientation_angular_velocity(self):
        body_twist = Twist3.Rx(1.0)
        pose = SO3.Rz(pi / 2)

        world_twist = body_twist3_to_world_twist3(pose, body_twist)

        self.assertTrue(np.allclose(world_twist.S, Twist3.Ry(-1.0).S))


class TestWorldTwist3ToBodyTwist3(unittest.TestCase):

    def test_identity_pose(self):
        world_twist = Twist3.Rand()
        pose = SO3()

        body_twist = world_twist3_to_body_twist3(pose, world_twist)

        self.assertTrue(np.allclose(body_twist.S, world_twist.S))

    def test_yaw_orientation_linear_velocity(self):
        world_twist = Twist3.Tx(1.0)
        pose = SO3.Rz(pi / 2)

        body_twist = world_twist3_to_body_twist3(pose, world_twist)

        self.assertTrue(np.allclose(body_twist.S, Twist3.Ty(1.0).S))

    def test_yaw_orientation_angular_velocity(self):
        world_twist = Twist3.Rx(1.0)
        pose = SO3.Rz(pi / 2)

        body_twist = world_twist3_to_body_twist3(pose, world_twist)

        self.assertTrue(np.allclose(body_twist.S, Twist3.Ry(1.0).S))


class TestFlattenSE3(unittest.TestCase):

    def test_se2(self):
        pose = SE2.Rand().SE3()
        flat_pose = flatten_se3(pose)

        self.assertTrue(np.allclose(flat_pose, pose))

    def test_se3(self):
        pose = SE3.Rand()
        flat_pose = flatten_se3(pose)
        _, _, y = pose.rpy(order='zyx')
        flat_r, flat_p, flat_y = flat_pose.rpy(order='zyx')

        self.assertTrue(np.allclose(flat_pose.t, pose.t))
        self.assertTrue(np.isclose(flat_r, 0))
        self.assertTrue(np.isclose(flat_p, 0))
        self.assertTrue(np.isclose(flat_y, y))


class TestFlattenTwist3(unittest.TestCase):

    def test_twist2(self):
        twist = Twist3(np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 3.0]))
        flat_twist = flatten_twist3(twist)

        self.assertTrue(np.allclose(flat_twist.S, twist.S))

    def test_twist3(self):
        twist = Twist3(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        flat_twist = flatten_twist3(twist)

        self.assertTrue(np.allclose(flat_twist.S, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
