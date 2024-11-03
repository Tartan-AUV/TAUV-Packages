import unittest
import numpy as np
from spatialmath import SE3, SO3, Twist3
import matplotlib.pyplot as plt

from trajectories.constant_acceleration_trajectory import ConstantAccelerationTrajectory, ConstantAccelerationTrajectoryParams
from trajectories.trajectory import Trajectory

def plot_trajectory(traj: Trajectory, name: str):
    n_points = 100

    times = np.linspace(-1, traj.duration + 1, n_points)

    poses = []
    twists = []

    position = np.zeros((n_points, 3))
    orientation = np.zeros((n_points, 3))
    linear_velocity = np.zeros((n_points, 3))
    angular_velocity = np.zeros((n_points, 3))

    for i, time in enumerate(times):
        pose, twist = traj.evaluate(time)
        poses.append(pose)
        twists.append(twist)

        position[i] = pose.t
        orientation[i] = pose.rpy()

        linear_velocity[i] = twist.v
        angular_velocity[i] = twist.w

    fig, axs = plt.subplots(4)
    fig.suptitle(name)

    axs[0].plot(times, position, label=["x", "y", "z"])
    axs[0].set_title("position")
    axs[0].legend()

    axs[1].plot(times, orientation, label=["roll", "pitch", "yaw"])
    axs[1].set_title("orientation")
    axs[1].legend()

    axs[2].plot(times, linear_velocity, label=["vx", "vy", "vz"])
    axs[2].set_title("linear_velocity")
    axs[2].legend()

    axs[3].plot(times, angular_velocity, label=["vax", "vay", "vaz"])
    axs[3].set_title("angular_velocity")
    axs[3].legend()

    plt.show()


class TestContantAccelerationTrajectory(unittest.TestCase):

    def test_positive_displacement_null_velocity(self):
        start_pose = SE3()
        end_pose = SE3.RPY(0, 0, 1.57) @ SE3.Tx(1.0)
        start_twist = Twist3()
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'positive_displacement_null_velocity')

    def test_negative_displacement_null_velocity(self):
        start_pose = SE3()
        end_pose = SE3.RPY(0, 0, -1.57)
        start_twist = Twist3()
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'negative_displacement_null_velocity')

    def test_positive_positive(self):
        start_pose = SE3.Tz(3.081)
        end_pose = SE3.Tz(1)
        start_twist = Twist3(np.array([-5.0586e-05, -0.0002554, 0.00069794]), np.array([0, 0, 0]))
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'positive_positive')

    def test_orthogonal_linear_velocity(self):
        start_pose = SE3.Tx(-1.0) @ SE3.Ty(1e-4)
        end_pose = SE3.Tx(1.0) @ SE3.Ty(2e-4)
        start_twist = Twist3.Ty(1.0) + Twist3.Tz(1.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'orthogonal_linear_velocity')

    def test_parallel_linear_velocity(self):
        start_pose = SE3()
        end_pose = SE3.Tx(1.0)
        start_twist = Twist3.Tx(1.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'parallel_linear_velocity')

    def test_orthogonal_angular_velocity(self):
        start_pose = SE3()
        end_pose = SE3.Rz(1.57)
        start_twist = Twist3.Rx(1.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'orthogonal_angular_velocity')

    def test_parallel_angular_velocity(self):
        start_pose = SE3()
        end_pose = SE3.Rz(1.57)
        start_twist = Twist3.Rz(1.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'parallel_angular_velocity')

    def test_excessive_linear_velocity(self):
        start_pose = SE3()
        end_pose = SE3.Tx(1.0)
        start_twist = Twist3.Tx(2.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'excessive_linear_velocity')

    def test_excessive_angular_velocity(self):
        start_pose = SE3()
        end_pose = SE3.Rz(1.57)
        start_twist = Twist3.Rz(2.0)
        end_twist = Twist3()

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=1.0,
            a_linear=1.0,
            v_max_angular=1.0,
            a_angular=1.0,
        )

        traj = ConstantAccelerationTrajectory(
            start_pose, start_twist,
            end_pose, end_twist,
            params
        )

        plot_trajectory(traj, 'excessive_angular_velocity')


if __name__ == "__main__":
    unittest.main()