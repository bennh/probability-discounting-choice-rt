import unittest

import numpy as np

from pd_project.valuation_choice import (
    choice_logits,
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
        for invalid in (np.nan, np.inf, 0.0, -1.0):
            with self.assertRaises(ValueError):
                parameter_by_condition(np.array(["R", "L"]), invalid, 1.0)
        with self.assertRaisesRegex(ValueError, "scalars"):
            parameter_by_condition(np.array(["R", "L"]), [1.0], 1.0)

    def test_only_scalar_inputs_are_broadcast(self):
        values = subjective_values(
            r_cert=np.array([10.0, 20.0]),
            r_uncert=np.array([20.0, 40.0]),
            odds=1.0,
            k=2.0,
            s0=10.0,
        )
        self.assertEqual(values.delta_v.shape, (2,))

        with self.assertRaisesRegex(ValueError, "same shape"):
            subjective_values(
                r_cert=np.array([[10.0], [20.0]]),
                r_uncert=np.array([20.0, 40.0]),
                odds=1.0,
                k=2.0,
                s0=10.0,
            )

    def test_choice_logits_direction_and_validation(self):
        logits = choice_logits(2.0, np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(logits, np.array([-2.0, 0.0, 2.0]))
        for invalid in (np.nan, np.inf, 0.0, -1.0):
            with self.assertRaises(ValueError):
                choice_logits(invalid, np.array([0.0]))

    def test_choice_probability_is_stable(self):
        probability = choice_probability(np.array([-1.0e6, 0.0, 1.0e6]))
        self.assertEqual(probability[0], 0.0)
        self.assertAlmostEqual(probability[1], 0.5)
        self.assertEqual(probability[2], 1.0)


if __name__ == "__main__":
    unittest.main()
