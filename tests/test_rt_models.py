import unittest

import numpy as np

from pd_project.rt_models import rt_location, rt_predictor


class RTModelTests(unittest.TestCase):
    def test_three_frozen_predictors(self):
        v_cert = np.array([1.0, -1.0, 2.0])
        v_uncert = np.array([1.5, -0.5, -3.0])
        delta_v = np.array([0.5, 0.5, -5.0])
        np.testing.assert_allclose(
            rt_predictor("M1", v_cert, v_uncert, delta_v), [0.5, 0.5, 5.0]
        )
        np.testing.assert_allclose(
            rt_predictor("M2", v_cert, v_uncert, delta_v), [0.25, 0.25, 25.0]
        )
        np.testing.assert_allclose(
            rt_predictor("M3", v_cert, v_uncert, delta_v), [1.25, 0.75, 2.5]
        )

    def test_rt_condition_location(self):
        location = rt_location(
            alpha=0.6,
            delta_loss=0.2,
            b=0.3,
            condition=np.array(["R", "L"]),
            predictor=np.array([2.0, 2.0]),
        )
        np.testing.assert_allclose(location, np.array([0.0, 0.2]))

    def test_invalid_model_and_overflow_fail(self):
        with self.assertRaises(ValueError):
            rt_predictor("M4", [1.0], [2.0], [1.0])
        with self.assertRaises(FloatingPointError):
            rt_predictor("M2", [0.0], [1.0e308], [1.0e308])
        with self.assertRaises(ValueError):
            rt_location(0.0, 0.0, 0.0, ["R"], [1.0])

    def test_predictor_requires_aligned_consistent_values(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            rt_predictor(
                "M1",
                np.array([[1.0], [2.0]]),
                np.array([2.0, 3.0]),
                np.array([1.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "delta_v must equal"):
            rt_predictor("M1", [1.0, 2.0], [2.0, 3.0], [99.0, 99.0])

    def test_location_requires_scalar_parameters_and_nonnegative_signal(self):
        with self.assertRaisesRegex(ValueError, "alpha must be a scalar"):
            rt_location([0.0], 0.0, 1.0, ["R"], [1.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            rt_location(0.0, 0.0, 1.0, ["R"], [-1.0])
        with self.assertRaisesRegex(ValueError, "same shape"):
            rt_location(
                0.0,
                0.0,
                1.0,
                np.array([["R"], ["L"]]),
                np.array([1.0, 2.0]),
            )
        with self.assertRaises(FloatingPointError):
            rt_location(1.0e308, 0.0, 1.0e308, ["R"], [1.0e308])


if __name__ == "__main__":
    unittest.main()
