import unittest
import numpy as np
from OptimalSpline import OptimalSpline


class SplineTest(unittest.TestCase):
    def test_at_waypoints(self):
        c = np.array([[-1, 3, 0, 2], [-1, 0, 3, 4], [-1, -3, 0, 6]]).transpose()
        ts = [0, 1, 2, 3]
        s = OptimalSpline(c, ts)
        self.assertEqual(s.val(0, 0), 2)
        self.assertEqual(s.val(0, 1), 4)
        self.assertEqual(s.val(0, 2), 6)
        self.assertEqual(s.val(0, 3), 2)

    def test_interpolation(self):
        c = np.array([[-1, 3, 0, 2], [-1, 0, 3, 4], [-1, -3, 0, 6]]).transpose()
        ts = [0, 1, 2, 3]
        s = OptimalSpline(c, ts)
        self.assertAlmostEqual(s.val(0, 4), -14)
        self.assertAlmostEqual(s.val(0, -1), 6)
        self.assertAlmostEqual(s.val(0, 0.8), 3.408)
        self.assertAlmostEqual(s.val(0, 2.8), 3.568)

    def test_derivatives(self):
        c = np.array([[-1, 3, 0, 2], [-1, 0, 3, 4], [-1, -3, 0, 6]]).transpose()
        ts = [0, 1, 2, 3]
        s = OptimalSpline(c, ts)
        self.assertEqual(s.val(1, 0), 0)
        self.assertEqual(s.val(1, 1), 3)
        self.assertEqual(s.val(1, 2), 0)
        self.assertEqual(s.val(1, 3), -9)
        self.assertAlmostEqual(s.val(1, 0.5), 2.25)
        self.assertAlmostEqual(s.val(1, 2.5), -3.75)
        self.assertEqual(s.val(2, 0), 6)
        self.assertEqual(s.val(2, 1), 0)
        self.assertEqual(s.val(2, 2), -6)
        self.assertEqual(s.val(2, 3), -12)
        self.assertAlmostEqual(s.val(2, 0.5), 3)
        self.assertAlmostEqual(s.val(2, 2.5), -9)
        self.assertEqual(s.val(3, 0), -6)
        self.assertEqual(s.val(3, 1), -6)
        self.assertEqual(s.val(3, 2), -6)
        self.assertEqual(s.val(3, 3), -6)
        self.assertEqual(s.val(3, 0.5), -6)
        self.assertEqual(s.val(3, 2.5), -6)

    def test_coefficients(self):
        num_segments = 4
        order = 5
        ts = [0, 1, 3, 4, 5]

        coefficients = np.random.rand(num_segments * (order+1))
        c = np.fliplr(coefficients.reshape((num_segments, order+1)))
        s = OptimalSpline(c.transpose(), ts)
        self.assertTrue((s._get_coeff_vector() == coefficients).all())



if __name__ == '__main__':
    unittest.main()
