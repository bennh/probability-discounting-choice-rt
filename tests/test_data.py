import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pd_project.data import (
    audit_trials,
    choice_mask,
    encode_choice_uncertain,
    load_data_directory,
    load_participant_mat,
    odds_against,
    prepare_trial_matrix,
    probability_percent_to_unit,
    rt_mask,
)


COLUMNS = {
    "r_cert": 0,
    "r_uncert": 1,
    "probability_percent": 2,
    "raw_action": 3,
    "p_cert": 4,
    "condition_code": 5,
    "rt_seconds": 6,
    "odds_source": 7,
}

LABELS = {
    "r_cert": "certOutcome",
    "r_uncert": "uncOutcome",
    "probability_percent": "outcome prob.",
    "raw_action": "action (1=certain, 2=uncertain, 0=missing)",
    "p_cert": "p_cert",
    "condition_code": "condition (1=reward,2=loss)",
    "rt_seconds": "RT",
    "odds_source": "odds",
}

CODING = {
    "missing_action": 0,
    "certain_action": 1,
    "uncertain_action": 2,
    "reward_condition": 1,
    "loss_condition": 2,
}

TRANSFORMS = {
    "probability_divisor": 100.0,
    "rt_divisor": 1000.0,
    "odds_assert_atol": 1.0e-9,
}


def raw_matrix() -> np.ndarray:
    return np.array(
        [
            [10, 20, 75, 1, np.nan, 1, 2000, 0.333333333],
            [-10, -20, 90, 2, np.nan, 2, 2500, 0.111111111],
        ],
        dtype=float,
    )


def small_trials() -> pd.DataFrame:
    probability = np.array([0.75, 0.50, 0.25, 0.90])
    raw_action = np.array([1.0, 2.0, 0.0, 1.0])
    rt_seconds = np.array([2.0, 2.1, np.nan, 2.2])
    return pd.DataFrame(
        {
            "participant": ["p1"] * 4,
            "run": ["A", "A", "B", "B"],
            "trial_index": [1, 2, 1, 2],
            "condition": ["R", "L", "R", "L"],
            "r_cert": [10.0, -10.0, 10.0, -10.0],
            "r_uncert": [20.0, -20.0, 20.0, -20.0],
            "probability": probability,
            "odds": odds_against(probability),
            "raw_action": raw_action,
            "choice_uncertain": encode_choice_uncertain(raw_action),
            "rt_seconds": rt_seconds,
            "choice_included": choice_mask(raw_action),
            "rt_included": rt_mask(raw_action, rt_seconds),
        }
    )


def small_data_config() -> dict:
    return {
        "coding": copy.deepcopy(CODING),
        "transforms": copy.deepcopy(TRANSFORMS),
        "expected": {
            "participants": 1,
            "run_values": ["A", "B"],
            "condition_values": ["R", "L"],
            "trial_index_start": 1,
            "trials_per_participant_run": 2,
            "trials_per_condition_per_participant_run": 1,
            "total_trials": 4,
            "missing_choice_trials": 1,
            "valid_choice_trials": 3,
            "valid_rt_trials": 3,
            "raw_action_values": [0, 1, 2],
            "missing_action_code_trials": 1,
            "nan_raw_action_trials": 0,
            "nonfinite_amount_values": 0,
            "nonfinite_probability_values": 0,
            "nonfinite_odds_values": 0,
            "probability_levels_unit": [0.25, 0.50, 0.75, 0.90],
            "odds_mismatch_trials": 0,
            "choice_encoding_mismatch_trials": 0,
            "choice_mask_definition_mismatch_trials": 0,
            "rt_mask_definition_mismatch_trials": 0,
            "choice_rt_mask_mismatch_trials": 0,
            "missing_action_with_observed_rt_trials": 0,
            "reward_amounts_positive": True,
            "loss_amounts_negative": True,
        },
    }


def prepare(matrix: np.ndarray, **overrides) -> pd.DataFrame:
    arguments = {
        "participant": "p1",
        "run": "A",
        "columns": copy.deepcopy(COLUMNS),
        "coding": copy.deepcopy(CODING),
        "odds_assert_atol": 1.0e-9,
        "probability_divisor": 100.0,
        "rt_divisor": 1000.0,
    }
    arguments.update(overrides)
    return prepare_trial_matrix(matrix, **arguments)


class DataContractTests(unittest.TestCase):
    def test_action_encoding_preserves_input_and_explicit_low_level_codes(self):
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

        configured = encode_choice_uncertain(
            [0, 5, 6], certain_action=5, uncertain_action=6
        )
        np.testing.assert_allclose(configured, [np.nan, 0.0, 1.0], equal_nan=True)

    def test_choice_and_rt_masks_are_independently_derived(self):
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
        np.testing.assert_allclose(
            odds_against(probability), np.array([9, 3, 1, 1 / 3, 1 / 9])
        )

    def test_invalid_probabilities_fail_loudly(self):
        for invalid in ([0.0], [100.0], [-1.0], [np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probability_percent_to_unit(invalid)
        for invalid in ([0.0], [1.0], [-0.1], [np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                odds_against(invalid)

    def test_rounded_source_odds_and_rt_unit_conversion(self):
        tidy = prepare(raw_matrix())
        np.testing.assert_allclose(tidy["odds"], [1 / 3, 1 / 9])
        np.testing.assert_allclose(tidy["rt_seconds"], [2.0, 2.5])
        self.assertTrue(tidy["choice_included"].all())
        self.assertTrue(tidy["rt_included"].all())

        wrong = raw_matrix()
        wrong[0, COLUMNS["odds_source"]] = 0.34
        with self.assertRaisesRegex(ValueError, "Source odds disagree"):
            prepare(wrong)

    def test_source_odds_and_tolerance_must_be_finite(self):
        for invalid_source in (np.nan, np.inf, -np.inf):
            matrix = raw_matrix()
            matrix[0, COLUMNS["odds_source"]] = invalid_source
            with self.subTest(source=invalid_source), self.assertRaises(ValueError):
                prepare(matrix)
        for invalid_tolerance in (np.nan, np.inf, -1.0):
            with self.subTest(tolerance=invalid_tolerance), self.assertRaises(ValueError):
                prepare(raw_matrix(), odds_assert_atol=invalid_tolerance)

        infinite_rt = raw_matrix()
        infinite_rt[0, COLUMNS["rt_seconds"]] = np.inf
        with self.assertRaisesRegex(ValueError, "RT cannot contain infinite"):
            prepare(infinite_rt)

    def test_matrix_shape_and_column_mapping_are_strict(self):
        with self.assertRaises(ValueError):
            prepare(np.empty((0, 8)))
        with self.assertRaisesRegex(ValueError, "transposed"):
            prepare(np.ones((8, 200)))
        with self.assertRaises(ValueError):
            prepare(np.ones((2, 9)))

        invalid_mappings = []
        negative = copy.deepcopy(COLUMNS)
        negative["raw_action"] = -1
        invalid_mappings.append(negative)
        fractional = copy.deepcopy(COLUMNS)
        fractional["raw_action"] = 3.5
        invalid_mappings.append(fractional)
        duplicate = copy.deepcopy(COLUMNS)
        duplicate["r_uncert"] = 0
        invalid_mappings.append(duplicate)
        missing = copy.deepcopy(COLUMNS)
        missing.pop("p_cert")
        invalid_mappings.append(missing)
        for mapping in invalid_mappings:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                prepare(raw_matrix(), columns=mapping)

    def test_unknown_finite_action_is_rejected_by_loader(self):
        matrix = raw_matrix()
        matrix[0, COLUMNS["raw_action"]] = 3
        with self.assertRaisesRegex(ValueError, "Unknown finite action"):
            prepare(matrix)

    def test_audit_accepts_complete_contract(self):
        audit = audit_trials(small_trials(), small_data_config())
        self.assertTrue(audit["passed_integrity_checks"])
        self.assertTrue(audit["passed_frozen_contract"])
        self.assertTrue(audit["approved_for_fitting"])
        self.assertEqual(audit["deviations"], {})

    def test_audit_recomputes_derived_fields(self):
        mutations = {
            "choice_encoding_mismatch_trials": ("choice_uncertain", 0, 1.0),
            "choice_mask_definition_mismatch_trials": ("choice_included", 0, False),
            "rt_mask_definition_mismatch_trials": ("rt_included", 0, False),
            "odds_mismatch_trials": ("odds", 0, 999.0),
        }
        for expected_key, (column, row, value) in mutations.items():
            trials = small_trials()
            trials.loc[row, column] = value
            audit = audit_trials(trials, small_data_config())
            with self.subTest(field=column):
                self.assertFalse(audit["passed_frozen_contract"])
                self.assertEqual(audit[expected_key], 1)
                self.assertIn(expected_key, audit["integrity_deviations"])

    def test_audit_requires_exact_grid_and_trial_identity(self):
        invalid_run = small_trials()
        invalid_run.loc[invalid_run["run"] == "B", "run"] = "C"
        audit = audit_trials(invalid_run, small_data_config())
        self.assertIn("run_values", audit["integrity_deviations"])

        invalid_condition = small_trials()
        invalid_condition.loc[invalid_condition["condition"] == "L", "condition"] = "X"
        audit = audit_trials(invalid_condition, small_data_config())
        self.assertIn("condition_values", audit["integrity_deviations"])

        duplicate_trial = small_trials()
        duplicate_trial.loc[1, "trial_index"] = 1
        audit = audit_trials(duplicate_trial, small_data_config())
        self.assertIn("duplicate_participant_run_trial", audit["integrity_deviations"])

    def test_audit_rejects_nan_probability_and_ambiguous_mask_dtype(self):
        trials = small_trials()
        trials.loc[0, "probability"] = np.nan
        audit = audit_trials(trials, small_data_config())
        self.assertIn("nonfinite_probability_values", audit["integrity_deviations"])

        trials = small_trials()
        trials["choice_included"] = trials["choice_included"].map(
            {True: "True", False: "False"}
        )
        with self.assertRaisesRegex(ValueError, "boolean"):
            audit_trials(trials, small_data_config())

        trials = small_trials()
        trials.loc[2, "rt_seconds"] = np.inf
        audit = audit_trials(trials, small_data_config())
        self.assertIn("infinite_rt_values", audit["integrity_deviations"])

    def test_audit_rejects_whitespace_in_identifiers_and_labels(self):
        mutations = (
            ("participant", 0, " p1 ", "participant_whitespace"),
            ("run", 0, " A ", "run_condition_whitespace"),
            ("condition", 0, " R ", "run_condition_whitespace"),
        )
        for column, row, value, expected_key in mutations:
            trials = small_trials()
            trials.loc[row, column] = value
            audit = audit_trials(trials, small_data_config())
            with self.subTest(column=column):
                self.assertFalse(audit["passed_frozen_contract"])
                self.assertIn(expected_key, audit["integrity_deviations"])


class MatlabLoaderTests(unittest.TestCase):
    def data_config(self) -> dict:
        return {
            "matlab_keys": {
                "run_a": "data_train",
                "run_b": "data_test",
                "labels": "data_labels",
            },
            "columns": copy.deepcopy(COLUMNS),
            "column_labels": copy.deepcopy(LABELS),
            "coding": copy.deepcopy(CODING),
            "transforms": copy.deepcopy(TRANSFORMS),
        }

    def test_matlab_labels_must_match_exact_schema(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("SciPy is unavailable")

        labels_in_order = [LABELS[name] for name, _ in sorted(COLUMNS.items(), key=lambda x: x[1])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1.mat"
            savemat(
                path,
                {
                    "data_train": raw_matrix(),
                    "data_test": raw_matrix(),
                    "data_labels": np.asarray(labels_in_order, dtype=object),
                },
            )
            loaded = load_participant_mat(path, self.data_config())
            self.assertEqual(loaded.shape, (4, 13))

            wrong = self.data_config()
            wrong["column_labels"]["r_cert"] = "WRONG"
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                load_participant_mat(path, wrong)

    def test_local_real_dataset_contract_when_available(self):
        root = Path(__file__).resolve().parents[1]
        raw_files = sorted((root / "data" / "raw").glob("**/*.mat"))
        if not raw_files:
            self.skipTest("Local raw PD data are not present")
        try:
            from pd_project.config import load_config
            import yaml  # noqa: F401
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("Local YAML/SciPy runtime is unavailable")

        config = load_config(root / "config" / "analysis.yaml")
        trials = load_data_directory(root / "data" / "raw", config["data"])
        audit = audit_trials(trials, config["data"])
        self.assertEqual(trials.shape, (19600, 13))
        self.assertEqual(trials["participant"].nunique(), 49)
        self.assertTrue(audit["passed_frozen_contract"])
        run_a_rt = trials.loc[
            (trials["run"] == "A") & trials["rt_included"], "rt_seconds"
        ]
        np.testing.assert_allclose(
            np.quantile(run_a_rt, [0.005, 0.995]),
            [0.274574, 8.230152],
            atol=1.0e-9,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
