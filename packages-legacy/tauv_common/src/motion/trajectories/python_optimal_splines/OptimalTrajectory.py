from motion.trajectories.python_optimal_splines import OptimalSplineGen
import numpy as np
import scipy.optimize
from math import factorial, sqrt
from motion.trajectories.python_optimal_splines import OptimalMultiSplineGen


class OptimalTrajectory:
    def __init__(self, order, ndims, waypoints, min_derivative_order=4, continuity_order=2, constraint_check_dt=0.05):
        self.ndims = ndims
        self.order = order
        self.waypoints = waypoints
        self.solved = False
        self.splines = []
        self.min_derivative_order = min_derivative_order
        self.continuity_order = continuity_order
        self.num_segs = len(waypoints) - 1
        self.constraint_check_dt = constraint_check_dt
        self.has_multispline_constraints = False
        assert self.num_segs >= 1

        for wp in waypoints:
            if len(wp.soft_directional_constraints) > 0:
                self.has_multispline_constraints = True
                break

    def solve(self, aggressiveness=0.1, time_opt_order=1, use_faster_ts=True, T=None, skip_times=False):
        self.use_faster_ts = use_faster_ts
        self._aggro = aggressiveness
        self._time_opt_order = time_opt_order

        minperseg = 50.0 / 1000
        maxperseg = 500.0  # TODO: this should NOT be necessary!

        if skip_times:
            self.splines = self._gen_splines()
            self.solved = True
            return

        if T is not None:
            if not use_faster_ts:
                print ("ERROR! Cannot use specified T unless use_faster_ts is enabled.")
                return None

        if T is None:
            if self.use_faster_ts:
                x0 = np.ones(1) * 100
                bounds = [(50.0 / 1000, 1000)]
            else:
                x0 = np.ones(self.num_segs) * 10
                bounds = [(minperseg, maxperseg) for i in range(self.num_segs)]

            res = scipy.optimize.minimize(
                self._cost_fn,
                x0,
                bounds=bounds,
                options={'disp': True,
                         'ftol': 1e-12,
                         'eps': 1e-4})
            x = res.x
        else:
            x = np.array([T])
        ts = self._arrange_ts(x)
        for i, wp in enumerate(self.waypoints):
            wp.set_time(ts[i])

        self.splines = self._gen_splines()
        self.solved = True

    def val(self, t, dim=None, order=0):
        if not self.solved:
            print("TRAJECTORY NOT SOLVED!!")
            return None

        if dim is None:
            return [s.val(order, t) for s in self.splines]

        return self.splines[dim].val(order, t)

    def end_time(self):
        return self.waypoints[-1].time

    def get_times(self):
        return [w.time for w in self.waypoints]

    def _arrange_ts(self, x):
        if self.use_faster_ts:
            dists = [None] * (len(self.waypoints) - 1)
            for i, wp in enumerate(self.waypoints[0:-1]):
                # Get position:
                pos = np.array(list(wp.get_pos()))
                next_pos = np.array(list(self.waypoints[i + 1].get_pos()))
                dists[i] = sqrt(np.dot((next_pos - pos), (next_pos - pos)))
            v = sum(dists) / x[0]
            return np.hstack((np.array([0]), np.cumsum([float(d) / v for d in dists])))
        else:
            return np.hstack((np.array([0]), np.cumsum(x)))

    def _cost_fn(self, x):
        return self._aggro * sum(x) ** 2 + self._compute_avg_cost_per_dim(x) ** 2

    def _compute_avg_cost_per_dim(self, x):
        ts = self._arrange_ts(x)
        for i, wp in enumerate(self.waypoints):
            wp.set_time(ts[i])

        if ts[-1] > 10000:
            print ("bad optimizer!")
            return 10000

        splines = self._gen_splines()
        if splines is None:
            print("no splines!")
            return 10000

        order = self.order
        num_segments = self.num_segs

        cw = order + 1  # constraint width
        x_dim = cw * num_segments  # num columns in the constraint matrix

        # Construct Hermitian matrix:
        H = np.zeros((x_dim, x_dim))
        for seg in range(0, num_segments):
            Q = self._compute_Q(order, self._time_opt_order, 0, ts[seg + 1] - ts[seg])  # x[seg])
            H[cw * seg:cw * (seg + 1), cw * seg:cw * (seg + 1)] = Q

        res = 0
        for spline in splines:
            try:
                c = spline._get_coeff_vector()
                res += c.dot(H.dot(c.transpose()))
            except:
                print('broken splines!')
                return 10000
        return res / self.ndims / ts[-1]

    def _compute_Q(self, order, min_derivative_order, t1, t2):
        r = min_derivative_order
        n = order

        T = np.zeros((n - r) * 2 + 1)
        for i in range(0, len(T)):
            T[i] = t2 ** (i + 1) - t1 ** (i + 1)

        Q = np.zeros((n + 1, n + 1))

        for i in range(r, n + 1):
            for j in range(i, n + 1):
                k1 = i - r
                k2 = j - r
                k = k1 + k2 + 1
                Q[i, j] = self._dc(k1, k1 + r) * self._dc(k2, k2 + r) / k * T[k - 1]
                Q[j, i] = Q[i, j]
        return Q

    def _calc_tvec(self, t, polynomial_order, tvec_order):
        r = tvec_order
        n = polynomial_order
        tvec = np.zeros(n + 1)
        for i in range(r, n + 1):
            tvec[i] = self._dc(i - r, i) * t ** (i - r)
        return tvec

    def _dc(self, d, p):
        return factorial(p) / factorial(d)

    def _gen_splines(self):
        if self.has_multispline_constraints:
            return OptimalMultiSplineGen.compute_min_derivative_multispline(self.order,
                                                                            self.min_derivative_order,
                                                                            self.continuity_order,
                                                                            self.waypoints)
        else:
            splines = [None] * self.ndims
            for i in range(self.ndims):
                pins = [wp.spline_pins[i] for wp in self.waypoints]
                splines[i] = OptimalSplineGen.compute_min_derivative_spline(self.order,
                                                                            self.min_derivative_order,
                                                                            self.continuity_order,
                                                                            pins)
                if splines[i] is None:
                    self.splines = None
                    return None
            return splines
