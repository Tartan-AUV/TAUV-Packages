import numpy as np

class RecursiveLeastSquares:
    def __init__(self, A_shape, b_cov=None, P_0=None, x_0=None) -> None:
        m, n = A_shape
        self._A_shape = A_shape

        self._b_cov = b_cov
        if self._b_cov is None:
            self._b_cov = np.zeros(shape=(m, m))
        
        self.P_0 = P_0
        if self.P_0 is None:
            