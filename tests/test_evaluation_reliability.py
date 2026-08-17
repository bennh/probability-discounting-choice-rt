import unittest

import numpy as np

import pandas as pd

from pd_project.evaluation import (
    holm_adjust,
    paired_participant_bootstrap,
    participant_cluster_bootstrap,
    participant_condition_means,
    support_shift_flags,
    trial_scores,
)
from pd_project.reliability import (
    bland_altman_summary,
    icc_a1,
    paired_bootstrap_icc_difference,
)


class EvaluationReliabilityTests(unittest.TestCase):
    def test_support_shift_flags(self):
        np.testing.assert_array_equal(
            support_shift_flags([0.0, 1.0, 2.0], [-0.1, 0.0, 1.5, 2.1]),
            [True, False, False, True],
        )

    def test_trial_scores_reject_rt_without_valid_choice(self):
        with self.assertRaises(ValueError):
            trial_scores(
                [np.nan],
                [1.0],
                [0.0],
                [0.0],
                1.0,
                choice_included=[False],
                rt_included=[True],
            )

    def test_trial_scores_require_boolean_masks(self):
        with self.assertRaises(ValueError):
            trial_scores(
                [1.0], [1.0], [0.0], [0.0], 1.0,
                choice_included=[1], rt_included=[0],
            )

    def test_trial_scores_ignore_nonfinite_predictors_on_excluded_trials(self):
        scores = trial_scores(
            [1.0, np.nan], [1.0, np.nan], [0.0, np.nan], [0.0, np.nan], 1.0,
            choice_included=np.array([True, False]),
            rt_included=np.array([True, False]),
        )
        self.assertTrue(np.isfinite(scores.loc[0, "choice_log_score"]))
        self.assertTrue(scores.loc[1].isna().all())

    def test_trial_scores_reject_scalar_inputs_and_invalid_sigma(self):
        with self.assertRaises(ValueError):
            trial_scores(
                1.0, 1.0, 0.0, 0.0, 1.0,
                choice_included=np.array(True), rt_included=np.array(True),
            )
        with self.assertRaises(ValueError):
            trial_scores(
                [np.nan], [np.nan], [np.nan], [np.nan], np.nan,
                choice_included=np.array([False]),
                rt_included=np.array([False]),
            )

    def test_participant_condition_uses_median_for_seconds_error(self):
        scored = pd.DataFrame(
            {
                "participant": ["p1", "p1", "p1"],
                "condition": ["R", "R", "R"],
                "model": ["M1", "M1", "M1"],
                "choice_log_score": [-1.0, -2.0, -3.0],
                "absolute_rt_error_seconds": [1.0, 2.0, 100.0],
            }
        )
        summary = participant_condition_means(
            scored, ["choice_log_score", "absolute_rt_error_seconds"]
        )
        self.assertAlmostEqual(summary.loc[0, "choice_log_score"], -2.0)
        self.assertAlmostEqual(summary.loc[0, "absolute_rt_error_seconds"], 2.0)

    def test_participant_condition_rejects_missing_group_labels(self):
        scored = pd.DataFrame(
            {"participant": ["p1", None], "condition": ["R", "R"],
             "choice_log_score": [-1.0, -2.0]}
        )
        with self.assertRaises(ValueError):
            participant_condition_means(scored, ["choice_log_score"])

    def test_paired_bootstrap_is_reproducible(self):
        first = paired_participant_bootstrap(
            ["p1", "p2", "p3"], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=100, seed=7
        )
        second = paired_participant_bootstrap(
            ["p1", "p2", "p3"], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=100, seed=7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 1.0)

    def test_paired_bootstrap_allows_repeated_participant_rows(self):
        result = paired_participant_bootstrap(
            ["p1", "p1", "p2", "p2"],
            [1.0, 3.0, 5.0, 7.0],
            [0.0, 2.0, 4.0, 6.0],
            n_boot=50,
            seed=11,
        )
        self.assertAlmostEqual(result["estimate"], 1.0)
        self.assertEqual(result["n_participants"], 2)

    def test_paired_bootstrap_rejects_invalid_contract(self):
        with self.assertRaises(ValueError):
            paired_participant_bootstrap(
                ["p1", None], [1.0, 2.0], [0.0, 1.0], n_boot=10, seed=1
            )
        with self.assertRaises(ValueError):
            paired_participant_bootstrap(
                ["p1", "p2"], [1.0, 2.0], [0.0, 1.0],
                n_boot=10, seed=1, confidence_level=1.0,
            )

    def test_cluster_bootstrap_carries_all_rows_for_an_id(self):
        frame = pd.DataFrame(
            {
                "participant": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "condition": ["R", "L"] * 3,
            }
        )

        def minimum_cluster_size(sample):
            return float(sample.groupby("participant").size().min())

        point, boot = participant_cluster_bootstrap(
            frame,
            participant_column="participant",
            statistic=minimum_cluster_size,
            n_boot=30,
            seed=4,
        )
        self.assertEqual(point, 2.0)
        np.testing.assert_allclose(boot, np.full(30, 2.0))

    def test_holm_adjustment_preserves_original_order(self):
        adjusted = holm_adjust([0.04, 0.01, 0.03])
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])

    def test_icc_and_bland_altman(self):
        measurements = np.column_stack((np.arange(1.0, 7.0), np.arange(1.0, 7.0)))
        self.assertAlmostEqual(icc_a1(measurements), 1.0)
        summary = bland_altman_summary(np.arange(4.0), np.arange(4.0) + 0.5)
        self.assertAlmostEqual(summary["mean_shift_b_minus_a"], 0.5)

    def test_icc_is_scale_invariant_for_small_values(self):
        measurements = np.column_stack((np.arange(1.0, 7.0), np.arange(1.0, 7.0)))
        self.assertAlmostEqual(icc_a1(measurements * 1.0e-5), 1.0)

    def test_icc_a1_known_absolute_agreement_value(self):
        measurements = np.array(
            [[9.0, 2.0], [6.0, 4.0], [8.0, 5.0], [7.0, 6.0], [10.0, 8.0]]
        )
        self.assertAlmostEqual(icc_a1(measurements), 0.12987012987012986)
        self.assertAlmostEqual(
            icc_a1(measurements * 1.0e-5 + 123.0),
            0.12987012987012986,
            places=8,
        )

    def test_paired_icc_bootstrap_is_keyed_and_row_order_invariant(self):
        frame = pd.DataFrame(
            {
                "participant": ["p3", "p1", "p5", "p2", "p4"],
                "full_a": [3.0, 1.0, 5.0, 2.0, 4.0],
                "full_b": [3.1, 1.2, 5.1, 1.9, 4.2],
                "choice_a": [2.0, 1.0, 4.0, 3.0, 5.0],
                "choice_b": [2.4, 0.8, 3.7, 3.5, 4.6],
            }
        )
        arguments = dict(
            participant_column="participant",
            full_a_column="full_a", full_b_column="full_b",
            choice_a_column="choice_a", choice_b_column="choice_b",
            n_boot=100, seed=42, minimum_valid_fraction=0.5,
        )
        first = paired_bootstrap_icc_difference(frame, **arguments)
        second = paired_bootstrap_icc_difference(
            frame.sample(frac=1.0, random_state=9), **arguments
        )
        self.assertEqual(first, second)
        self.assertEqual(first["n_complete_participants"], 5)

    def test_paired_icc_bootstrap_rejects_invalid_identifiers_and_count(self):
        frame = pd.DataFrame(
            {
                "participant": ["p1", "p2", None],
                "full_a": [1.0, 2.0, 3.0], "full_b": [1.0, 2.0, 3.0],
                "choice_a": [1.0, 2.0, 3.0], "choice_b": [1.0, 2.0, 3.0],
            }
        )
        arguments = dict(
            participant_column="participant",
            full_a_column="full_a", full_b_column="full_b",
            choice_a_column="choice_a", choice_b_column="choice_b",
            seed=1,
        )
        with self.assertRaises(ValueError):
            paired_bootstrap_icc_difference(frame, n_boot=10, **arguments)
        frame.loc[2, "participant"] = "p3"
        with self.assertRaises(ValueError):
            paired_bootstrap_icc_difference(frame, n_boot=0, **arguments)


if __name__ == "__main__":
    unittest.main()
