import unittest

import numpy as np

from pd_project.valuation_choice import (
    choice_probability,
    parameter_by_condition,
    subjective_values,
)


class ValuationChoiceTests(unittest.TestCase):
    def test_signed_reward_and_loss_values(self):
        values = subjective_values(
            r_cert=np.array([10.0, -10.0]),
            r_uncert=np.array([20.0, -20.0]),
            odds=np.array([1.0, 1.0]),
            k=np.array([3.0, 3.0]),
            s0=10.0,
        )
        np.testing.assert_allclose(values.v_cert, np.array([1.0, -1.0]))
        np.testing.assert_allclose(values.v_uncert, np.array([0.5, -0.5]))
        np.testing.assert_allclose(values.delta_v, np.array([-0.5, 0.5]))

    def test_condition_specific_parameters(self):
        result = parameter_by_condition(np.array(["R", "L", "R"]), 1.5, 3.0)
        np.testing.assert_allclose(result, np.array([1.5, 3.0, 1.5]))
        with self.assertRaises(ValueError):
            parameter_by_condition(np.array(["R", "unknown"]), 1.0, 1.0)

    def test_choice_probability_is_stable(self):
        probability = choice_probability(np.array([-1.0e6, 0.0, 1.0e6]))
        self.assertEqual(probability[0], 0.0)
        self.assertAlmostEqual(probability[1], 0.5)
        self.assertEqual(probability[2], 1.0)


if __name__ == "__main__":
    unittest.main()

