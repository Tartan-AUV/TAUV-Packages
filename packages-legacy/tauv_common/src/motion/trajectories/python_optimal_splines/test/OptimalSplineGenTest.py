import unittest
import OptimalSplineGen


class TestOptimalSplineGen(unittest.TestCase):
    def test_Q(self):
        Q = OptimalSplineGen._compute_Q(5, 3, 1, 2)
        self.assertEqual(Q[5, 3], 840)
        self.assertEqual(Q[5, 4], 5400)
        self.assertEqual(Q[5, 5], 22320)
        self.assertEqual(Q[3, 5], 840)
        self.assertEqual(Q[4, 5], 5400)
        self.assertEqual(Q[5, 5], 22320)

    def test_tvec(self):
        T = OptimalSplineGen._calc_tvec(3, 5, 3)
        self.assertEqual(T[0], 0)
        self.assertEqual(T[1], 0)
        self.assertEqual(T[2], 0)
        self.assertEqual(T[3], 6)
        self.assertEqual(T[4], 72)
        self.assertEqual(T[5], 540)

    def test_optimization(self):
        wp1 = OptimalSplineGen.Waypoint(2)
        wp1.add_hard_constraint(0, 4)
        wp1.add_hard_constraint(1, 2.5)

        wp2 = OptimalSplineGen.Waypoint(3)
        wp2.add_hard_constraint(0, 2)

        wp3 = OptimalSplineGen.Waypoint(6)
        wp3.add_hard_constraint(0, -1.7)
        wp3.add_hard_constraint(1, 0)

        wp4 = OptimalSplineGen.Waypoint(4)
        wp4.add_soft_constraint(0, 3, 0.2)

        wp5 = OptimalSplineGen.Waypoint(7)
        wp5.add_soft_constraint(0, 3, 0.2)
        wp6 = OptimalSplineGen.Waypoint(8)
        wp6.add_soft_constraint(0, 3, 0.2)

        s = OptimalSplineGen.compute_min_derivative_spline(5, 3, 2, [wp1, wp2, wp3, wp4, wp5, wp6])
        self.assertAlmostEqual(s.val(0, 2), 4, places=2)
        self.assertAlmostEqual(s.val(1, 2), 2.5, places=2)
        self.assertAlmostEqual(s.val(0, 3), 2, places=2)
        self.assertAlmostEqual(s.val(0, 6), -1.7, places=2)
        self.assertAlmostEqual(s.val(1, 6), 0, places=2)
        self.assertLessEqual(s.val(0, 4), 3.21)
        self.assertGreaterEqual(s.val(0, 4), 2.79)


if __name__ == '__main__':
    unittest.main()
