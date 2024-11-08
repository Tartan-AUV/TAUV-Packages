from motion.trajectories.python_optimal_splines.OptimalSplineGen import Waypoint


class TrajectoryWaypoint:
    def __init__(self, ndim):
        self.time = None
        self.pos = None
        if isinstance(ndim, int):
            self.ndim = ndim
            self.spline_pins = [Waypoint(None) for i in range(self.ndim)]
        elif isinstance(ndim, tuple):
            self.ndim = len(ndim)
            self.spline_pins = [Waypoint(None) for i in range(self.ndim)]
            self.add_hard_constraints(0, ndim)
        else:
            raise ValueError("no")
        self.soft_directional_constraints = []

    def add_hard_constraints(self, order, values):
        assert len(values) == self.ndim
        for i, v in enumerate(values):
            self.spline_pins[i].add_hard_constraint(order, v)
        if order == 0:
            self.pos = values

    def add_soft_constraints(self, order, values, radii):
        assert len(values) == self.ndim
        for i, v in enumerate(values):
            self.spline_pins[i].add_soft_constraint(order, v, radii[i])

    def add_hard_constraint(self, order, dim, value):
        self.spline_pins[dim].add_hard_constraint(order, value)

    def add_soft_constraint(self, order, dim, value, radius):
        self.spline_pins[dim].add_soft_constraint(order, value, radius)

    # TODO: support directional constraints that only use *some* of the dimensions
    def add_soft_directional_constraint(self, order, values, radius):
        self.soft_directional_constraints.append((order, values, radius))

    def add_hard_directional_constraint(self, order, values):
        self.add_soft_directional_constraint(order, values, 0)

    def set_time(self, t):
        self.time = t
        for sp in self.spline_pins:
            sp.time = t

    def get_pos(self):
        if self.pos is None:
            return None
        else:
            return self.pos