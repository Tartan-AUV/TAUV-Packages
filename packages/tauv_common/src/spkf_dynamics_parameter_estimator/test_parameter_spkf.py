import unittest

import numpy as np
import matplotlib.pyplot as plt

import dynamics
from parameter_spkf import ParameterSPKF

class ParameterSPKFTestCase(unittest.TestCase):

    def setUp(self):
        self.parameters = np.concatenate((
            np.array([30]),
            np.array([0.03]),
            np.zeros(3),
            np.zeros(3),
            10 * np.ones(6),
            np.ones(6),
            np.zeros(6),
            np.zeros(6)
        ))

        def measurement_function(state: np.array, input: np.array, parameters: np.array, error: np.array) -> np.array:
            return dynamics.get_acceleration(parameters, state, input) + error

        self.measurement_function = measurement_function

        self.initial_covariance = 1e-12 * np.identity(32)
        # Setting process covariance too high can cause it to explode
        self.process_covariance = 1e-4 * np.identity(32)
        self.error_covariance = 1e-4 * np.identity(6)

        self.parameter_limits = np.array([
            [0, 1e10],
            [0, 1e10],
            [-1e10, 1e10],
            [-1e10, 1e10],
            [-1e10, 1e10],
            [-1e10, 1e10],
            [-1e10, 1e10],
            [-1e10, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
            [0, 1e10],
        ])

    def test_zero_wrench_update(self):
        initial_parameters = np.concatenate((
            np.array([30]),
            np.array([0.03]),
            np.zeros(3),
            np.zeros(3),
            10 * np.ones(6),
            np.ones(6),
            np.ones(6),
            np.ones(6)
        ))

        spkf = ParameterSPKF(
            measurement_function=self.measurement_function,
            initial_parameters=initial_parameters,
            initial_covariance=self.initial_covariance,
            process_covariance=self.process_covariance,
            error_covariance=self.error_covariance,
            dim_parameters=32,
            dim_state=9,
            dim_input=6,
            dim_measurement=6,
            parameter_limits=self.parameter_limits
        )

        state = np.zeros(9)
        input = np.zeros(6)

        measurement = dynamics.get_acceleration(self.parameters, state, input)

        spkf.update(state, input, measurement)

        parameters = spkf.get_parameters()
        covariance = spkf.get_parameter_covariance()

        self.assertTrue(np.allclose(parameters, initial_parameters))

    def test_nonzero_wrench_update(self):
        initial_parameters = np.concatenate((
            np.array([30]),
            np.array([0.03]),
            np.zeros(3),
            np.zeros(3),
            10 * np.ones(6),
            np.ones(6),
            np.ones(6),
            np.ones(6)
        ))

        spkf = ParameterSPKF(
            measurement_function=self.measurement_function,
            initial_parameters=initial_parameters,
            initial_covariance=self.initial_covariance,
            process_covariance=self.process_covariance,
            error_covariance=self.error_covariance,
            dim_parameters=32,
            dim_state=9,
            dim_input=6,
            dim_measurement=6,
            parameter_limits=self.parameter_limits
        )

        state = np.zeros(9)
        input = np.array([30, 0, 0, 0, 0, 0])

        measurement = dynamics.get_acceleration(self.parameters, state, input)

        spkf.update(state, input, measurement)

        parameters = spkf.get_parameters()
        covariance = spkf.get_parameter_covariance()

        self.assertTrue(np.allclose(parameters, initial_parameters))

    def test_convergence(self):
        initial_parameters = np.concatenate((
            np.clip(np.array([30]) + np.random.normal(scale=1, size=1), 0, np.inf),
            np.clip(np.array([0.03]) + np.random.normal(scale=1e-2, size=1), 0, np.inf),
            np.zeros(3) + np.random.normal(scale=1e-2, size=3),
            np.zeros(3) + np.random.normal(scale=1e-2, size=3),
            np.clip(10 * np.ones(6) + np.random.normal(scale=1, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=1, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=1, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=1, size=6), 0, np.inf)
        ))

        spkf = ParameterSPKF(
            measurement_function=self.measurement_function,
            initial_parameters=initial_parameters,
            initial_covariance=self.initial_covariance,
            process_covariance=self.process_covariance,
            error_covariance=self.error_covariance,
            dim_parameters=32,
            dim_state=9,
            dim_input=6,
            dim_measurement=6,
            parameter_limits=self.parameter_limits
        )

        num_iterations = 500

        intermediate_parameters = np.zeros((num_iterations, 32))

        for i in range(num_iterations):
            state = np.concatenate((
                np.random.uniform(-np.pi, np.pi, size=3),
                np.random.normal(scale=1, size=6)
            ))
            input = np.random.normal(scale=50, size=6)

            measurement = dynamics.get_acceleration(self.parameters, state, input)

            spkf.update(state, input, measurement)

            intermediate_parameters[i, :] = spkf.get_parameters()

        plot_all_params(intermediate_parameters, self.parameters)

    def test_convergence_with_noise(self):
        initial_parameters = np.concatenate((
            np.clip(np.array([30]) + np.random.normal(scale=10, size=1), 0, np.inf),
            np.clip(np.array([0.03]) + np.random.normal(scale=1e-1, size=1), 0, np.inf),
            np.zeros(3) + np.random.normal(scale=1e-2, size=3),
            np.zeros(3) + np.random.normal(scale=1e-2, size=3),
            np.clip(10 * np.ones(6) + np.random.normal(scale=1, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=10, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=10, size=6), 0, np.inf),
            np.clip(np.ones(6) + np.random.normal(scale=10, size=6), 0, np.inf)
        ))

        spkf = ParameterSPKF(
            measurement_function=self.measurement_function,
            initial_parameters=initial_parameters,
            initial_covariance=self.initial_covariance,
            process_covariance=self.process_covariance,
            error_covariance=self.error_covariance,
            dim_parameters=32,
            dim_state=9,
            dim_input=6,
            dim_measurement=6,
            parameter_limits=self.parameter_limits
        )

        num_iterations = 500

        intermediate_parameters = np.zeros((num_iterations, 32))

        for i in range(num_iterations):
            state = np.concatenate((
                np.random.uniform(-np.pi, np.pi, size=3),
                np.random.normal(scale=1, size=6)
            ))
            input = np.random.normal(scale=50, size=6)

            measurement = dynamics.get_acceleration(self.parameters, state, input) + np.random.normal(scale=1e-2, size=6)

            spkf.update(state, input, measurement)

            intermediate_parameters[i, :] = spkf.get_parameters()

        plot_all_params(intermediate_parameters, self.parameters)

def plot_all_params(intermediate_parameters, parameters):
    params_fig, params_axs = plt.subplots(8, figsize=(8, 16))
    params_fig.suptitle("Parameter Evolution", fontsize=18)

    plot_params(params_axs[0], intermediate_parameters[:, 0], np.array([parameters[0]]), ["m"], "Mass")
    plot_params(params_axs[1], intermediate_parameters[:, 1], np.array([parameters[1]]), ["v"], "Volume")
    plot_params(params_axs[2], intermediate_parameters[:, 2:5], parameters[2:5], ["gx", "gy", "gz"],
                "Center of Gravity")
    plot_params(params_axs[3], intermediate_parameters[:, 5:8], parameters[5:8], ["bx", "by", "bz"],
                "Center of Buoyancy")
    plot_params(params_axs[4], intermediate_parameters[:, 8:14], parameters[8:14],
                ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"], "Inertia")
    plot_params(params_axs[5], intermediate_parameters[:, 14:20], parameters[14:20],
                ["du", "dv", "dw", "dp", "dq", "dr"],
                "Linear Damping")
    plot_params(params_axs[6], intermediate_parameters[:, 20:26], parameters[20:26],
                ["ddu", "ddv", "ddw", "ddp", "ddq", "ddr"], "Quadratic Damping")
    plot_params(params_axs[7], intermediate_parameters[:, 26:32], parameters[26:32],
                ["au", "av", "aw", "ap", "aq", "ar"],
                "Added Mass")

    params_fig.tight_layout()
    plt.show()

def plot_params(ax, intermediate_params, params, legend, title):
    ax.plot(intermediate_params)
    for (i, param) in enumerate(params):
        ax.axhline(param, linestyle='--')
    ax.grid()
    ax.legend(legend, ncol=len(legend), loc="upper center")
    ax.set_title(title)

if __name__ == '__main__':
    unittest.main()
