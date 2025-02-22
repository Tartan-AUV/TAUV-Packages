import unittest
from planners.mpc_planner.mpc import MPC
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_mpc(self):
        FLAT_STATES = 7
        FLAT_CTRLS = 4
        A = np.zeros((FLAT_STATES, FLAT_STATES))
        A[0:3, 3:6] = np.eye(3)
        B = np.zeros((FLAT_STATES, FLAT_CTRLS))
        B[3:, :] = np.eye(4)
        Q = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 10])
        R = np.eye(FLAT_CTRLS) * 1
        S = Q * 10

        mpc = MPC(A, B, Q, R, S, N=10, dt=0.1)

        x = np.zeros((FLAT_STATES, 1))
        x_ref = np.array([1, 1, 1, 0, 0, 0, 1])[np.newaxis].transpose()
        x_ref = np.repeat(x_ref, 11, 1)

        u, x = mpc.solve(x, x_ref)
        print(x)
        print(u)

    def test_mpc_double_integrator(self):
        T = 0.1
        A = np.zeros((2,2))
        A[0, 1] = 1
        B = np.array([T / 2, 1])[np.newaxis].transpose()
        Q = np.eye(2)
        R = np.eye(1)
        S = np.eye(2) * 10

        mpc = MPC(A, B, Q, R, S, N=10, dt=T)

        x = np.zeros((2, 1))
        x_ref = np.array([2, 0])[np.newaxis].transpose()
        x_ref = np.repeat(x_ref, 11, 1)

        u, x = mpc.solve(x, x_ref)
        # print(x)
        # print(u)


if __name__ == '__main__':
    unittest.main()
