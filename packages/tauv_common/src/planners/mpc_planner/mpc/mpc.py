# MPC.py
#
# Python implementation of a Model Predictive Controller (MPC). Suitable for linear
# mpc problems with linear constraints. Under the hood it uses the OSQP QP solver
# to compute the optimal control inputs.
#
# Ros imports are only necessary for generating Path objects for visualization in rviz,
# but could be removed.
#
# Author: Tom Scherlis
#


import osqp
import numpy as np
from scipy import linalg, sparse
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import Header
import rospy
from math import atan2, asin
from scipy.spatial.transform import Rotation


class MPC:
    def __init__(self, A, B, Q, R, S, N, dt, x_constraints=None, u_constraints=None):
        A = dt * A + np.eye(len(A))
        B = dt * B
        self.N = N
        num_nonconstrained = 1  # int(round(N / 2))
        self.dt = dt

        xdim = len(A)
        udim = B.shape[1]

        self.xdim = xdim
        self.udim = udim

        INF = 1e10

        x_constraints_inf = np.hstack((-INF * np.ones((xdim, 1)), INF * np.ones((xdim, 1))))
        if x_constraints is None:
            x_constraints = np.hstack((-INF * np.ones((xdim, 1)), INF * np.ones((xdim, 1))))
        if u_constraints is None:
            u_constraints = np.hstack((-INF * np.ones((udim, 1)), INF * np.ones((udim, 1))))

        L_x = np.vstack((np.eye(xdim), -np.eye(xdim)))
        b_x = np.vstack((x_constraints[:, 1][np.newaxis].transpose(), -x_constraints[:, 0][np.newaxis].transpose()))
        b_x_inf = np.vstack((x_constraints_inf[:, 1][np.newaxis].transpose(), -x_constraints_inf[:, 0][np.newaxis].transpose()))
        L_u = np.vstack((np.eye(udim), -np.eye(udim)))
        b_u = np.vstack((u_constraints[:, 1][np.newaxis].transpose(), -u_constraints[:, 0][np.newaxis].transpose()))

        self.L_u_bar = linalg.block_diag(*tuple([L_u] * N))
        self.L_x_bar = linalg.block_diag(*tuple([L_x] * (N + 1)))
        self.b_u_bar = np.vstack(tuple([b_u] * N))
        self.b_x_bar = np.vstack((np.vstack(tuple([b_x_inf] * (num_nonconstrained))), np.vstack(tuple([b_x] * (N - num_nonconstrained + 1)))))
        self.Q_bar = linalg.block_diag(*tuple([Q] * (N + 1)))
        self.Q_bar[-xdim - 1:-1, -xdim - 1:-1] = S
        self.R_bar = linalg.block_diag(*tuple([R] * N))
        self.A_bar = self._build_a_bar(A)
        self.B_bar = self._build_b_bar(A, B)

        self.m = None

    def solve(self, x, x_ref):
        assert x.shape[0] == self.xdim
        assert x.shape[1] == 1

        assert x_ref.shape[0] == self.xdim
        assert x_ref.shape[1] == self.N + 1

        f_bar = np.dot(self.A_bar, x)
        G_bar = x_ref.transpose().reshape((self.xdim * (self.N + 1), 1))
        C = np.dot(np.dot(f_bar.transpose(), self.Q_bar), self.B_bar) - np.dot(np.dot(G_bar.transpose(), self.Q_bar),
                                                                               self.B_bar)
        b = np.vstack((self.b_u_bar, self.b_x_bar - np.dot(self.L_x_bar, f_bar)))

        if self.m is None:
            H = np.dot(np.dot(self.B_bar.transpose(), self.Q_bar), self.B_bar) + self.R_bar
            L = np.vstack((self.L_u_bar, np.dot(self.L_x_bar, self.B_bar)))
            self.m = osqp.OSQP()
            self.m.setup(P=sparse.csc_matrix(H), q=C.transpose(), l=None, A=sparse.csc_matrix(L), u=b, verbose=False,
                         warm_start=True)
        else:
            self.m.update(q=C.transpose(), l=None, u=b)

        res = self.m.solve()
        if res.info.status_val != 1:
            rospy.logerr("[MPC Solver] Solver error: " + res.info.status)
            return None, None

        u = res.x
        x = f_bar + np.dot(self.B_bar, u[np.newaxis].transpose())

        u = u.reshape(self.N, self.udim).transpose()
        x = x.reshape(self.N + 1, self.xdim).transpose()

        return u, x

    def to_path(self, x, start_time, frame='odom'):
        assert x.shape[0] == self.xdim
        assert x.shape[0] == 8

        num_poses = x.shape[1]

        poses = []
        for i in range(num_poses):
            t = start_time + rospy.Duration(self.dt * i)
            p = PoseStamped()
            p.pose.position = Point(x=x[0, i], y=x[1, i], z=x[2, i])
            vel = np.array([x[4, i], x[5, i], x[6, i]])
            if abs(linalg.norm(vel)) > 1e-8:
                vel = vel / linalg.norm(vel)
            psi = atan2(vel[1], vel[0])
            theta = asin(-vel[2])
            q = Rotation.from_euler('ZYX', [psi, theta, 0]).as_quat()
            p.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            p.header.frame_id = frame
            p.header.stamp = t
            poses.append(p)

        path = Path()
        path.header.frame_id = frame
        path.header.stamp = start_time
        path.poses = poses
        return path

    def _build_a_bar(self, A):
        rm = A.shape[0]
        cm = A.shape[1]
        A_bar = np.zeros((rm * (self.N + 1), cm))
        for i in range(self.N + 1):
            A_bar[rm * i:rm * (i + 1), :] = np.linalg.matrix_power(A, i)
        return A_bar

    def _build_b_bar(self, A, B):
        rm = B.shape[0]
        cm = B.shape[1]
        B_bar = np.zeros((rm * (self.N + 1), cm * self.N))
        for r in range(self.N + 1):
            for c in range(self.N):
                order = r - c - 1
                if order < 0:
                    B_bar[rm * r:rm * (r + 1), cm * c:cm * (c + 1)] = np.zeros(B.shape)
                else:
                    B_bar[rm * r:rm * (r + 1), cm * c:cm * (c + 1)] = np.dot(np.linalg.matrix_power(A, order), B)
        return B_bar
