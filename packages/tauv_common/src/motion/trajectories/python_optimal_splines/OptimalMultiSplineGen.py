import numpy as np
from math import factorial
import osqp
from scipy import sparse, linalg
from motion.trajectories.python_optimal_splines.OptimalSpline import OptimalSpline


# This code is influenced by the implementation here: https://github.com/symao/minimum_snap_trajectory_generation,
# However, splines are implemented slightly differently.

class Waypoint:
    def __init__(self, time):
        self.hard_constraints = []  # (order, value)
        self.soft_constraints = []  # (order, value, radius)
        self.time = time

    def add_hard_constraint(self, order, value):
        self.hard_constraints.append((order, value))

    def add_soft_constraint(self, order, value, radius):
        self.soft_constraints.append((order, value, radius))


def compute_min_derivative_multispline(order, min_derivative_order, continuity_order, traj_waypoints):
    num_segments = len(traj_waypoints) - 1
    if num_segments < 2:
        return None

    cw = order + 1  # constraint width
    x_dim = cw * num_segments  # num columns in the constraint matrix
    multi_x_dim = x_dim * traj_waypoints[0].ndim

    Aeq = np.zeros((0, 0))  # equality constraints A matrix
    Aieq = np.zeros((0, 0))  # inequality constraints A matrix
    beq = np.zeros((0, 1))
    bieq = np.zeros((0, 1))
    H = np.zeros((0, 0))

    for dim in range(traj_waypoints[0].ndim):
        pins = [wp.spline_pins[dim] for wp in traj_waypoints]
        dAeq, dbeq, dAieq, dbieq, dH = compute_spline_matrices(order, min_derivative_order, continuity_order, pins)

        Aeq = linalg.block_diag(Aeq, dAeq)
        Aieq = linalg.block_diag(Aieq, dAieq)
        H = linalg.block_diag(H, dH)
        beq = np.vstack((beq, dbeq))
        bieq = np.vstack((bieq, dbieq))

    # build directional constraints (constraints between spline dimensions)
    for seg in range(num_segments):
        wp = traj_waypoints[seg]
        for sdc in wp.soft_directional_constraints:
            con_order = sdc[0]
            dvec = sdc[1]
            radius = sdc[2]

            dspace = np.zeros((wp.ndim, wp.ndim))
            dspace[0,:] = np.array(dvec)
            nspace = linalg.null_space(dspace)
            nullspace_vecs = list(nspace.transpose())

            # Positive dot product enforces correct direction of motion
            new_constraint = np.zeros((1, multi_x_dim))
            for d in range(wp.ndim):
                scalar = dvec[d]
                tvec = _calc_tvec(0, order, con_order)
                vec = scalar * tvec
                new_constraint[0, x_dim * d + seg * cw: x_dim * d + (seg + 1) * cw] = vec
            Aieq = np.vstack((Aieq, -1 * new_constraint))
            bieq = np.vstack((bieq, 0))

            # Limit motion in null space:
            for v in nullspace_vecs:
                new_constraint = np.zeros((1, multi_x_dim))
                for d in range(wp.ndim):
                    scalar = v[d]
                    tvec = _calc_tvec(0, order, con_order)
                    vec = scalar * tvec
                    new_constraint[0, x_dim * d + seg * cw: x_dim * d + (seg + 1) * cw] = vec
                Aieq = np.vstack((Aieq, new_constraint))
                Aieq = np.vstack((Aieq, -1*new_constraint))
                bieq = np.vstack((bieq, radius))
                bieq = np.vstack((bieq, -radius))

    # Convert equality constraints to inequality constraints:
    Aeq_ieq = np.vstack((Aeq, -1 * Aeq))
    beq_ieq = np.vstack((beq, -1 * beq))
    Aieq = np.vstack((Aieq, Aeq_ieq))
    bieq = np.vstack((bieq, beq_ieq))

    # Solve the QP!
    m = osqp.OSQP()
    try:
        m.setup(P=sparse.csc_matrix(H), q=None, l=None, A=sparse.csc_matrix(Aieq), u=bieq, verbose=True)
    except ValueError as ve:
        print(ve.message)
        print("Could not setup QP!")
        return None
    results = m.solve()
    x = results.x

    splines = []
    xwidth = len(x) // traj_waypoints[0].ndim
    for dim in range(traj_waypoints[0].ndim):
        dx = x[dim * xwidth: dim * xwidth + xwidth]
        coefficients = np.fliplr(np.array(dx).reshape((num_segments, order + 1)))
        ts = [wp.time for wp in traj_waypoints]
        spline = OptimalSpline(coefficients.transpose(), ts)
        splines.append(spline)

    return splines


def compute_spline_matrices(order, min_derivative_order, continuity_order, waypoints):
    num_segments = len(waypoints) - 1
    if num_segments < 2:
        return None

    assert not any([wp.time is None for wp in waypoints])

    waypoints.sort(key=lambda waypoint: waypoint.time)

    durations = [0] * num_segments
    for i in range(len(durations)):
        durations[i] = waypoints[i + 1].time - waypoints[i].time

    # Start with waypoint constraints: (Start of each segment, end of last segment.)
    # x is vertical stack of the coefficient col vectors for each segment
    cw = order + 1  # constraint width
    x_dim = cw * (num_segments)  # num columns in the constraint matrix

    Aeq = np.zeros((0, x_dim))  # equality constraints A matrix
    Aieq = np.zeros((0, x_dim))  # inequality constraints A matrix
    beq = np.zeros((0, 1))
    bieq = np.zeros((0, 1))

    # Build constraint matrices:
    for seg, wp in enumerate(waypoints):
        for con in wp.hard_constraints:
            if seg == num_segments:
                tvec = _calc_tvec(durations[seg - 1], order, con[0])
                i = seg - 1
            else:
                tvec = _calc_tvec(0, order, con[0])
                i = seg

            constraint = np.zeros(x_dim)  # hard constraint at waypoint
            constraint[i * cw: (i + 1) * cw] = tvec
            Aeq = np.vstack((Aeq, constraint))
            beq = np.vstack((beq, con[1]))

        for con in wp.soft_constraints:  # voxel constraints around waypoint
            if seg == num_segments:
                tvec = _calc_tvec(durations[seg - 1], order, con[0])
                i = seg - 1
            else:
                tvec = _calc_tvec(0, order, con[0])
                i = seg

            constraint_max = np.zeros(x_dim)
            constraint_max[i * cw: (i + 1) * cw] = tvec

            constraint_min = np.zeros(x_dim)
            constraint_min[i * cw: (i + 1) * cw] = -1 * tvec

            Aieq = np.vstack((Aieq, constraint_max, constraint_min))
            bieq = np.vstack((bieq, con[1] + con[2], -(con[1] - con[2])))

    # Continuity constraints: (tvec_a(t=end) = tvec_b(t=0))
    for seg in range(0, num_segments - 1):
        for r in range(0, continuity_order + 1):
            constraint = np.zeros(x_dim)  # hard constraint at waypoint
            tvec_end = _calc_tvec(durations[seg], order, r)
            tvec_start = _calc_tvec(0, order, r)
            constraint[seg * cw: (seg + 1) * cw] = tvec_end
            constraint[(seg + 1) * cw: (seg + 2) * cw] = -tvec_start
            Aeq = np.vstack((Aeq, constraint))
            beq = np.vstack((beq, [0]))

    # Construct Hermitian matrix:
    H = np.zeros((x_dim, x_dim))
    for seg in range(0, num_segments):
        Q = _compute_Q(order, min_derivative_order, 0, durations[seg])
        H[cw * seg:cw * (seg + 1), cw * seg:cw * (seg + 1)] = Q

    return Aeq, beq, Aieq, bieq, H


def _compute_Q(order, min_derivative_order, t1, t2):
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
            Q[i, j] = _dc(k1, k1 + r) * _dc(k2, k2 + r) / k * T[k - 1]
            Q[j, i] = Q[i, j]
    return Q


def _calc_tvec(t, polynomial_order, tvec_order):
    r = tvec_order
    n = polynomial_order
    tvec = np.zeros(n + 1)
    for i in range(r, n + 1):
        tvec[i] = _dc(i - r, i) * t ** (i - r)
    return tvec


def _dc(d, p):
    return factorial(p) / factorial(d)

