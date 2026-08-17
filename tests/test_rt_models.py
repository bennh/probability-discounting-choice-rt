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
            rt_predictor("M2", [0.0], [0.0], [1.0e308])
        with self.assertRaises(ValueError):
            rt_location(0.0, 0.0, 0.0, ["R"], [1.0])


if __name__ == "__main__":
    unittest.main()

