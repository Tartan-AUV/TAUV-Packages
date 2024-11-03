import numpy as np
from math import factorial

class OptimalSpline:
    def __init__(self, coefficients, ts):
        assert (isinstance(coefficients, np.ndarray))
        assert (len(ts) == coefficients.shape[1] + 1)
        self.coefficients = coefficients
        self.order = coefficients.shape[0] - 1
        self.num_segments = coefficients.shape[1]
        self.ts = ts

    def val(self, r, t):
        # Find appropriate segment:

        seg = 0
        while self.ts[seg] <= t:
            seg += 1
            if seg > self.num_segments - 1:
                break
        seg -= 1
        seg = max(0, seg)

        # Grab the coefficients for that segment
        coeffs = self.coefficients[:, seg]
        T = t - self.ts[seg]

        # evaluate the spline at that point:
        # return sum([coeffs[(self.order - i)] * T ** i for i in range(0, self.order + 1)])
        res = 0
        for i in range(r, self.order + 1):
            res += coeffs[(self.order - i)] * factorial(i) / factorial(i - r) * T**(i-r)
        return res

    def _get_coeff_vector(self):
        return np.fliplr(self.coefficients.transpose()).ravel()

    def __getitem__(self, t):
        return self.val(0, t)
