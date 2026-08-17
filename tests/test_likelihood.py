import unittest

import numpy as np

from pd_project.likelihood import (
    bernoulli_logpmf_from_logits,
    joint_log_likelihood,
    lognormal_logpdf,
)


class LikelihoodTests(unittest.TestCase):
    def test_extreme_choice_logits_remain_finite(self):
        logits = np.array([-1.0e6, 1.0e6, 1.0e6, -1.0e6])
        choice = np.array([0.0, 1.0, 0.0, 1.0])
        score = bernoulli_logpmf_from_logits(choice, logits)
        self.assertTrue(np.all(np.isfinite(score)))
        np.testing.assert_allclose(score[:2], np.zeros(2), atol=1.0e-12)
        np.testing.assert_allclose(score[2:], np.full(2, -1.0e6), rtol=0.0, atol=1.0e-6)

    def test_lognormal_density_includes_seconds_jacobian(self):
        observed = lognormal_logpdf(np.e, 0.0, 1.0)
        expected = -1.0 - 0.5 * np.log(2.0 * np.pi) - 0.5
        self.assertAlmostEqual(float(observed), float(expected))

    def test_joint_likelihood_uses_independent_masks(self):
        choice = np.array([0.0, 1.0, 0.0])
        rt = np.array([1.0, np.nan, 1.0])
        logits = np.zeros(3)
        mu = np.zeros(3)
        choice_include = np.array([True, True, True])
        rt_include = np.array([True, False, True])
        observed = joint_log_likelihood(
            choice,
            rt,
            logits,
            mu,
            1.0,
            choice_mask=choice_include,
            rt_mask=rt_include,
        )
        expected = 3 * -np.log(2.0) + 2 * (-0.5 * np.log(2.0 * np.pi))
        self.assertAlmostEqual(observed, expected)

    def test_rt_mask_must_be_subset_of_choice_mask(self):
        with self.assertRaises(ValueError):
            joint_log_likelihood(
                [np.nan],
                [1.0],
                [0.0],
                [0.0],
                1.0,
                choice_mask=[False],
                rt_mask=[True],
            )

    def test_only_scalar_likelihood_inputs_are_broadcast(self):
        observed = bernoulli_logpmf_from_logits([0.0, 1.0], 0.0)
        np.testing.assert_allclose(observed, np.full(2, -np.log(2.0)))
        with self.assertRaisesRegex(ValueError, "same shape"):
            bernoulli_logpmf_from_logits(
                np.array([[0.0], [1.0]]), np.array([0.0, 0.0])
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            lognormal_logpdf(
                np.array([[1.0], [2.0]]), np.array([0.0, 0.0]), 1.0
            )

    def test_joint_likelihood_requires_boolean_masks_and_scalar_sigma(self):
        arguments = dict(
            choice_uncertain=[0.0],
            rt_seconds=[1.0],
            logits=[0.0],
            mu=[0.0],
            sigma=1.0,
            choice_mask=np.array([True]),
            rt_mask=np.array([True]),
        )
        for name, invalid in (
            ("choice_mask", np.array([1])),
            ("rt_mask", np.array(["False"])),
        ):
            malformed = dict(arguments)
            malformed[name] = invalid
            with self.assertRaisesRegex(ValueError, "boolean dtype"):
                joint_log_likelihood(**malformed)

        nonscalar_sigma = dict(arguments)
        nonscalar_sigma["sigma"] = np.array([1.0])
        with self.assertRaisesRegex(ValueError, "sigma must be a scalar"):
            joint_log_likelihood(**nonscalar_sigma)

    def test_lognormal_numeric_overflow_fails_loudly(self):
        with self.assertRaises(FloatingPointError):
            lognormal_logpdf(
                rt_seconds=np.finfo(float).max,
                mu=-np.finfo(float).max,
                sigma=np.finfo(float).tiny,
            )


if __name__ == "__main__":
    unittest.main()
