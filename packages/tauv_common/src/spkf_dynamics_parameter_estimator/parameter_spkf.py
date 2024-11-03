import numpy as np
import scipy as sp
from math import sqrt

# Implementation based on
# ECE5550, SIMULTANEOUS STATE AND PARAMETER ESTIMATION course notes, page 9-11

class ParameterSPKF:

    def __init__(self,
                 measurement_function,
                 initial_parameters,
                 initial_covariance,
                 process_covariance,
                 error_covariance,
                 dim_parameters,
                 dim_state,
                 dim_input,
                 dim_measurement,
                 parameter_limits,
                 ):
        assert(initial_parameters.shape == (dim_parameters, ))
        assert(initial_covariance.shape == (dim_parameters, dim_parameters))
        assert(process_covariance.shape == (dim_parameters, dim_parameters))
        assert(error_covariance.shape == (dim_measurement, dim_measurement))
        assert(parameter_limits is None or parameter_limits.shape == (dim_parameters, 2))

        self._h = measurement_function

        self._theta = np.array(initial_parameters, dtype=np.float64)
        self._cov_theta = np.array(initial_covariance, dtype=np.float64)
        self._initial_cov_theta = np.array(initial_covariance, dtype=np.float64)
        self._cov_r = np.array(process_covariance, dtype=np.float64)
        self._cov_e = np.array(error_covariance, dtype=np.float64)

        self._dim_theta = dim_parameters
        self._dim_theta_a = dim_parameters + dim_measurement
        self._dim_x = dim_state
        self._dim_u = dim_input
        self._dim_d = dim_measurement

        if parameter_limits is not None:
            self._theta_limits = parameter_limits
        else:
            self._theta_limits = np.hstack((
                np.full(dim_parameters, -np.inf),
                np.full(dim_parameters, np.inf)
            ))

        self._gamma = sqrt(3)
        self._alpha_0_m = (self._gamma ** 2 - self._dim_theta_a) / (self._gamma ** 2)
        self._alpha_k_m = 1 / (2 * (self._gamma ** 2))
        self._alpha_0_c = self._alpha_0_m
        self._alpha_k_c = self._alpha_k_m

    def update(self, state, input, measurement):
        assert(state.shape == (self._dim_x, ))
        assert(input.shape == (self._dim_u, ))
        assert(measurement.shape == (self._dim_d, ))

        x = np.array(state, dtype=np.float64)
        u = np.array(input, dtype=np.float64)
        d = np.array(measurement, dtype=np.float64)

        theta = self._theta
        cov_theta = self._cov_theta

        theta_a = np.concatenate((
            theta,
            np.zeros(self._dim_d)
        ))

        cov_theta = cov_theta + self._cov_r

        cov_theta_a = sp.linalg.block_diag(self._cov_theta, self._cov_e)

        try:
            sqrt_cov_theta_a = np.linalg.cholesky(cov_theta_a)
        except np.linalg.LinAlgError:
            self._cov_theta = self._initial_cov_theta
            return

        W_a_deltas = np.hstack((
            np.zeros((self._dim_theta + self._dim_d, 1)),
            sqrt_cov_theta_a,
            -sqrt_cov_theta_a
        ))

        W_a = np.transpose(theta_a + self._gamma * np.transpose(W_a_deltas))

        D = np.zeros((self._dim_d, W_a.shape[1]))

        for i in range(D.shape[1]):
            W_theta_i = W_a[0:self._dim_theta, i]
            W_e_i = W_a[self._dim_theta:self._dim_theta + self._dim_d, i]

            D_i = self._h(x, u, W_theta_i, W_e_i)

            D[:, i] = D_i

        d_hat = self._alpha_0_m * D[:, 0]
        for i in range(1, D.shape[1]):
            d_hat = d_hat + self._alpha_k_m * D[:, i]

        cov_d = self._alpha_0_c * np.outer(D[:, 0] - d_hat, D[:, 0] - d_hat)
        for i in range(1, D.shape[1]):
            cov_d = cov_d + self._alpha_k_c * np.outer(D[:, i] - d_hat, D[:, i] - d_hat)

        cov_theta_d = self._alpha_0_c * np.outer(W_a[0:self._dim_theta, 0] - self._theta, D[:, 0] - d_hat)
        for i in range(1, D.shape[1]):
            cov_theta_d = cov_theta_d + self._alpha_k_c * np.outer(W_a[0:self._dim_theta, i] - self._theta, D[:, i] - d_hat)

        L = cov_theta_d @ np.linalg.inv(cov_d)

        theta = theta + L @ (d - d_hat)
        theta = np.clip(theta, self._theta_limits[:, 0], self._theta_limits[:, 1])
        cov_theta = cov_theta - L @ cov_d @ np.transpose(L)

        self._theta = theta
        self._cov_theta = cov_theta

        return self._theta, self._cov_theta

    def get_parameters(self):
        return self._theta

    def set_parameters(self, parameters):
        assert(parameters.shape == self._theta.shape)
        self._theta = parameters

    def get_parameter_covariance(self):
        return self._cov_theta

    def set_parameter_covariance(self, covariance):
        assert(covariance.shape == self._cov_theta.shape)
        self._cov_theta = covariance
