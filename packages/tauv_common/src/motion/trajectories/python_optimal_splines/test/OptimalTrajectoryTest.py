import unittest
import OptimalSplineGen
from OptimalTrajectory import TrajectoryWaypoint, OptimalTrajectory
import numpy as np


class TestOptimalSplineGen(unittest.TestCase):
    # def test_waypoints(self):
    #     tw = TrajectoryWaypoint(3)
    #     tw.add_hard_constraints(0, (1, 2, 3))
    #     self.assertEqual(len(tw.spline_pins), 3)
    #     self.assertEqual(len(tw.spline_pins[0]), 1)
    #     self.assertEqual(tw.spline_pins[0][0].hard_constraints[0], (0, 1))
    #     self.assertEqual(tw.spline_pins[1][0].hard_constraints[0], (0, 2))
    #     self.assertEqual(tw.spline_pins[2][0].hard_constraints[0], (0, 3))

    # def test_gen_splines(self):
    #     tw1 = TrajectoryWaypoint((1, 2, 3))
    #     tw1.add_hard_constraints(1, (0, 0, 0))
    #     tw2 = TrajectoryWaypoint((2, 0, 5))
    #     tw3 = TrajectoryWaypoint((1, 4, 7))
    #     tw3.add_hard_constraints(1, (0, 0, 0))
    #
    #     tw1.set_time(0)
    #     tw2.set_time(2)
    #     tw3.set_time(3)
    #
    #     wpts = [tw1, tw2, tw3]
    #     ot = OptimalTrajectory(5, 3, wpts)
    #     splines = ot._gen_splines()
    #     self.assertAlmostEqual(splines[0].val(0, 0), 1, places=2)
    #     self.assertAlmostEqual(splines[0].val(0, 2), 2, places=2)
    #     self.assertAlmostEqual(splines[0].val(0, 3), 1, places=2)
    #     self.assertAlmostEqual(splines[1].val(0, 0), 2, places=2)
    #     self.assertAlmostEqual(splines[1].val(0, 2), 0, places=2)
    #     self.assertAlmostEqual(splines[1].val(0, 3), 4, places=2)
    #     self.assertAlmostEqual(splines[2].val(0, 0), 3, places=2)
    #     self.assertAlmostEqual(splines[2].val(0, 2), 5, places=2)
    #     self.assertAlmostEqual(splines[2].val(0, 3), 7, places=2)
    #     self.assertAlmostEqual(splines[0].val(1, 0), 0, places=2)
    #     self.assertAlmostEqual(splines[0].val(1, 3), 0, places=2)
    #     self.assertAlmostEqual(splines[2].val(1, 0), 0, places=2)
    #     self.assertAlmostEqual(splines[2].val(1, 3), 0, places=2)
    #
    # def test_solve(self):
    #     tw1 = TrajectoryWaypoint((1, 2, 3))
    #     tw1.add_hard_constraints(1, (0, 0, 0))
    #     tw2 = TrajectoryWaypoint((2, 0, 4))
    #     tw3 = TrajectoryWaypoint((2, 4, 3))
    #     tw4 = TrajectoryWaypoint((4, 4, 5))
    #     tw5 = TrajectoryWaypoint((40, 23, 24))
    #     tw6 = TrajectoryWaypoint((4, 2, 5))
    #     tw7 = TrajectoryWaypoint((2, 6, 3))
    #     tw7.add_hard_constraints(1, (0, 0, 0))
    #
    #     wpts = [tw1, tw2, tw3, tw4, tw5, tw6, tw7]
    #     ot = OptimalTrajectory(5, 3, wpts)
    #     ot.solve()

    # def test_solve_directional_constraints(self):
    #     tw1 = TrajectoryWaypoint((1, 2, 3))
    #     tw1.add_hard_constraints(1, (0, 0, 0))
    #     tw2 = TrajectoryWaypoint((2, 0, 4))
    #     tw3 = TrajectoryWaypoint((2, 4, 3))
    #     tw4 = TrajectoryWaypoint((4, 4, 5))
    #     tw5 = TrajectoryWaypoint((40, 23, 24))
    #     tw6 = TrajectoryWaypoint((4, 2, 5))
    #     tw6.add_hard_directional_constraint(1, (1, 0, 0))
    #     tw7 = TrajectoryWaypoint((2, 6, 3))
    #     tw7.add_hard_constraints(1, (0, 0, 0))
    #
    #     wpts = [tw1, tw2, tw3, tw4, tw5, tw6, tw7]
    #     ot = OptimalTrajectory(5, 3, wpts)
    #     ot.solve()
    #
    #     t = ot.splines[0].ts[5]
    #     self.assertAlmostEqual(ot.val(t, 1, 1), 0, places=1)
    #     self.assertAlmostEqual(ot.val(t, 2, 1), 0, places=1)

    def test_solve_directional_constraints_harder(self):
        tw1 = TrajectoryWaypoint((1, 2, 3))
        tw1.add_hard_constraints(1, (0, 0, 0))
        tw2 = TrajectoryWaypoint((2, 0, 4))
        tw3 = TrajectoryWaypoint((2, 4, 3))
        tw4 = TrajectoryWaypoint((4, 4, 5))
        tw5 = TrajectoryWaypoint((4, 5, 3))
        tw6 = TrajectoryWaypoint((3, 4, 2))
        tw6.add_hard_directional_constraint(1, (-1, -1, -1))
        tw7 = TrajectoryWaypoint((2, 6, 3))
        tw7.add_hard_constraints(1, (0, 0, 0))

        wpts = [tw1, tw2, tw3, tw4, tw5, tw6, tw7]
        ot = OptimalTrajectory(5, 3, wpts)
        ot.solve()

        print(ot.splines[0].ts)
        t = ot.splines[0].ts[5]
        print(ot.val(t, None, 1))


if __name__ == '__main__':
    unittest.main()
