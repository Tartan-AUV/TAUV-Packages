import unittest
from dynamics.dynamics import Dynamics
from math import pi
import numpy as np

class TestDynamics(unittest.TestCase):
    def test_controller(self):
        dyn = Dynamics()

        eta = []
        v = []
        eta_dd = []

        tau_n = dyn.compute_tau(eta, v, eta_dd)

    def test_get_torque(self):
        eta = [0, 0, 0, 0, 0, pi/2]
        v = [0, 0, 0, 0, 0, 0]
        eta_dd = [1, 0, 0, 0, 0, 0]
        dyn = Dynamics()

        eta_d = dyn.get_eta_d(eta, v)

        tau_n = dyn.compute_tau_n(eta, eta_d, eta_dd)
        tau = dyn.compute_tau(eta, eta_d, eta_dd)
        print("tau: ", list(tau.flatten()), " tau_n: ", list(tau_n.flatten()))

    def test_second_order(self):
        eta = [0, 0, 0, 0, 0, pi/2]
        v = [0, 0, 0, 0, 0, 0]
        v_d = [1, 0, 0, 0, 0, 0]

        dyn = Dynamics()

        eta_d = dyn.get_eta_d(eta, v)
        print(dyn.get_eta_dd(eta, eta_d, v_d))


if __name__ == '__main__':
    unittest.main()
