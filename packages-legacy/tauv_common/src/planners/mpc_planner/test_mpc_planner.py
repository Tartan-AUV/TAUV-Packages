import unittest
import rospy
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, Twist, Vector3

from motion import trajectories
from tauv_msgs.srv import GetTrajRequest, GetTrajResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import rpy_to_quat

from planners.mpc_planner.mpc_planner import MPCPlanner


class TestMPCPlanner(unittest.TestCase):

    def setUp(self):
        rospy.init_node('mpc_planner')
        self.mpc_planner = MPCPlanner()

    def test_yaw_traj(self):
        self.mpc_planner._pose = Pose(
            position=tm([0.001, 0.001, 0.001], Vector3),
            orientation=rpy_to_quat([0.001, 0.001, 0.001])
        )

        self.mpc_planner._twist = Twist(
            linear=tm([0, 0, 0], Vector3),
            angular=tm([0, 0, 0], Vector3)
        )

        dest_pose = Pose(
            position=tm([0, 0, 0], Vector3),
            orientation=rpy_to_quat([0, 0, 1.57])
        )

        dest_twist = Twist(
            linear=tm([0, 0, 0], Vector3),
            angular=tm([0, 0, 0], Vector3)
        )

        traj = trajectories.NewLinearTrajectory(
            self.mpc_planner._pose,
            self.mpc_planner._twist,
            dest_pose,
            dest_twist
        )

        traj_request = GetTrajRequest()
        traj_request.len = 101
        traj_request.dt = 0.1
        traj_request.curr_pose = self.mpc_planner._pose
        traj_request.curr_twist = self.mpc_planner._twist
        traj_request.curr_time = traj.start_time

        traj_response = traj.get_points(traj_request)

        ref_traj, u_mpc, x_mpc, u = self.mpc_planner._update(None, traj_response)

        x = np.linspace(0, 10, 101)
        plt.plot(x, ref_traj[5,:])
        plt.plot(x, x_mpc[5,:])
        x = np.linspace(0, 9.9, 100)
        plt.plot(x, u_mpc[5,:])
        plt.show()


if __name__ == '__main__':
    unittest.main()
