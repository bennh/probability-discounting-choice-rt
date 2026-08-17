import unittest

import numpy as np
import pandas as pd

from pd_project.data import (
    audit_trials,
    choice_mask,
    encode_choice_uncertain,
    odds_against,
    probability_percent_to_unit,
    rt_mask,
)


class DataContractTests(unittest.TestCase):
    def test_action_encoding_preserves_certain_choices_and_input(self):
        raw = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
        original = raw.copy()
        encoded = encode_choice_uncertain(raw)
        np.testing.assert_allclose(
            encoded, np.array([np.nan, 0.0, 1.0, np.nan, np.nan]), equal_nan=True
        )
        np.testing.assert_allclose(raw, original, equal_nan=True)
        np.testing.assert_array_equal(
            choice_mask(raw), np.array([False, True, True, False, False])
        )

    def test_choice_and_rt_masks_are_independent(self):
        raw = np.array([1, 2, 0, 2, 1, 2], dtype=float)
        rt = np.array([1.0, np.nan, 1.0, 0.0, np.inf, 2.0])
        np.testing.assert_array_equal(
            choice_mask(raw), np.array([True, True, False, True, True, True])
        )
        np.testing.assert_array_equal(
            rt_mask(raw, rt), np.array([True, False, False, False, False, True])
        )

    def test_probability_and_odds_conversion(self):
        percent = np.array([10, 25, 50, 75, 90], dtype=float)
        probability = probability_percent_to_unit(percent)
        np.testing.assert_allclose(probability, np.array([0.10, 0.25, 0.50, 0.75, 0.90]))
        np.testing.assert_allclose(odds_against(probability), np.array([9, 3, 1, 1 / 3, 1 / 9]))

    def test_invalid_probabilities_fail_loudly(self):
        for invalid in ([0.0], [100.0], [-1.0], [np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probability_percent_to_unit(invalid)
        for invalid in ([0.0], [1.0], [-0.1], [np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                odds_against(invalid)

    def test_audit_detects_participant_run_imbalances(self):
        trials = pd.DataFrame(
            {
                "participant": ["p1"] * 4 + ["p2"] * 4,
                "run": ["A", "B", "B", "B", "A", "A", "A", "B"],
                "condition": ["R"] * 8,
                "choice_included": [True] * 8,
                "rt_included": [True] * 8,
            }
        )
        audit = audit_trials(
            trials,
            {
                "expected": {
                    "participants": 2,
                    "total_trials": 8,
                    "missing_choice_trials": 0,
                    "valid_choice_trials": 8,
                    "valid_rt_trials": 8,
                    "trials_per_participant_run": 2,
                }
            },
        )
        self.assertFalse(audit["passed_expected_counts"])
        self.assertEqual(
            len(audit["deviations"]["trials_per_participant_run"]["violations"]), 4
        )


if __name__ == "__main__":
    unittest.main()
