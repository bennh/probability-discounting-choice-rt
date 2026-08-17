import copy
from pathlib import Path
import unittest

import pandas as pd

from pd_project.config import load_config
from pd_project.fit import validate_run_a_fit_artifact


class RunAFitArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "config" / "analysis.yaml")

    def valid_frame(self) -> pd.DataFrame:
        common = {
            "participant": "p1",
            "success": True,
            "objective": 10.0,
            "log_k_R": 0.0,
            "log_k_L": 0.0,
            "log_beta": 0.0,
            "alpha": 0.0,
            "delta_L": 0.0,
            "log_b": 0.0,
            "log_sigma": -1.0,
        }
        return pd.DataFrame(
            [
                {**common, "model": "choice_only", "choice_only": True},
                {**common, "model": "M1", "choice_only": False},
                {**common, "model": "M2", "choice_only": False},
                {**common, "model": "M3", "choice_only": False},
            ]
        )

    def test_valid_matrix_is_partitioned(self):
        full, baseline = validate_run_a_fit_artifact(
            self.valid_frame(), ["p1"], self.config
        )
        self.assertEqual(set(full["model"]), {"M1", "M2", "M3"})
        self.assertEqual(baseline["model"].tolist(), ["choice_only"])

    def test_missing_bad_or_misclassified_parameters_fail_before_scoring(self):
        missing = self.valid_frame().drop(columns="log_sigma")
        with self.assertRaisesRegex(ValueError, "missing parameters"):
            validate_run_a_fit_artifact(missing, ["p1"], self.config)

        nonnumeric = self.valid_frame()
        nonnumeric["log_k_R"] = nonnumeric["log_k_R"].astype(object)
        nonnumeric.loc[nonnumeric["model"] == "M1", "log_k_R"] = "bad"
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_run_a_fit_artifact(nonnumeric, ["p1"], self.config)

        misclassified = self.valid_frame()
        misclassified.loc[misclassified["model"] == "choice_only", "choice_only"] = False
        with self.assertRaisesRegex(ValueError, "choice_only flag"):
            validate_run_a_fit_artifact(misclassified, ["p1"], self.config)

        out_of_bounds = copy.deepcopy(self.valid_frame())
        out_of_bounds.loc[out_of_bounds["model"] == "M2", "log_b"] = 100.0
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_run_a_fit_artifact(out_of_bounds, ["p1"], self.config)


if __name__ == "__main__":
    unittest.main()
