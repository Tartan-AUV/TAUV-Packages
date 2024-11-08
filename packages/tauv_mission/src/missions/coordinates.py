class Coord:
    def __init__(self, x=0, y=0, z=0, deg=None):
        self.x = x
        self.y = y
        self.z = z
        self.deg = deg

class Course:
    def __init__(self, _wall_t_rear, _rear_t_course, _wall_t_ref, _ref_t_gate, 
                 _ref_t_buoy, _buoy_t_approach, _ref_t_octagon,
                 _octagon_t_approach, _ref_t_torpedo, _torpedo_t_approach, 
                 _ref_t_marker, _marker_t_approach, _gate_t_style, _torpedo_approach_t_backoff,
                 _buoy_approach_t_backoff, _marker_approach_t_backoff, _octagon_approach_t_backoff):
        self._wall_t_rear = Coord(**_wall_t_rear)
        self._rear_t_course = Coord(**_rear_t_course)
        self._wall_t_ref = Coord(**_wall_t_ref)
        self._ref_t_gate = Coord(**_ref_t_gate)
        self._ref_t_buoy = Coord(**_ref_t_buoy)
        self._buoy_t_approach = Coord(**_buoy_t_approach)
        self._ref_t_octagon = Coord(**_ref_t_octagon)
        self._octagon_t_approach = Coord(**_octagon_t_approach)
        self._ref_t_torpedo = Coord(**_ref_t_torpedo)
        self._torpedo_t_approach = Coord(**_torpedo_t_approach)
        self._ref_t_marker = Coord(**_ref_t_marker)
        self._marker_t_approach = Coord(**_marker_t_approach)
        self._gate_t_style = Coord(**_marker_t_approach)
        self._torpedo_approach_t_backoff = Coord(**_torpedo_approach_t_backoff)
        self._buoy_approach_t_backoff = Coord(**_buoy_approach_t_backoff)
        self._marker_approach_t_backoff = Coord(**_marker_approach_t_backoff)
        self._octagon_approach_t_backoff = Coord(**_octagon_approach_t_backoff)
