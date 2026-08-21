import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_b_once.py"
SPEC = importlib.util.spec_from_file_location("run_b_once_script", SCRIPT)
run_b_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_b_once)


class RunBOnceContractTests(unittest.TestCase):
    def score_fixture(self):
        run_rows = []
        score_rows = []
        for participant in ("p1", "p2"):
            for trial, condition, choice_mask, rt_mask in (
                (1, "R", True, True),
                (2, "L", False, False),
            ):
                run_rows.append(
                    {
                        "participant": participant,
                        "trial_index": trial,
                        "condition": condition,
                        "choice_included": choice_mask,
                        "rt_included": rt_mask,
                    }
                )
                for model in ("choice_only", "M1", "M2", "M3"):
                    full = model != "choice_only"
                    score_rows.append(
                        {
                            "participant": participant,
                            "model": model,
                            "trial_index": trial,
                            "condition": condition,
                            "choice_log_score": -1.0 if choice_mask else np.nan,
                            "brier_score": 0.2 if choice_mask else np.nan,
                            "choice_correct": 1.0 if choice_mask else np.nan,
                            "rt_log_score": -1.0 if full and rt_mask else np.nan,
                            "absolute_log_rt_error": 0.1
                            if full and rt_mask
                            else np.nan,
                            "absolute_rt_error_seconds": 0.2
                            if full and rt_mask
                            else np.nan,
                            "out_of_support": False if full else np.nan,
                        }
                    )
        return pd.DataFrame(score_rows), pd.DataFrame(run_rows)

    def test_score_artifact_contract_accepts_exact_matrix(self):
        scores, run_b = self.score_fixture()
        run_b_once.validate_score_artifact(scores, run_b, ["p1", "p2"])

    def test_score_artifact_contract_rejects_mask_disagreement(self):
        scores, run_b = self.score_fixture()
        scores.loc[
            (scores["participant"] == "p1")
            & (scores["model"] == "M1")
            & (scores["trial_index"] == 2),
            "rt_log_score",
        ] = -2.0
        with self.assertRaises(RuntimeError):
            run_b_once.validate_score_artifact(scores, run_b, ["p1", "p2"])

    def test_reliability_contract_accepts_exact_matrix(self):
        results = [
            SimpleNamespace(
                participant=participant,
                model=model,
                success=False,
            )
            for participant in ("p1", "p2")
            for model in ("choice_only", "M1", "M2", "M3")
        ]

        frame = pd.DataFrame(
            {
                "participant": [result.participant for result in results],
                "model": [result.model for result in results],
                "success": [result.success for result in results],
            }
        )

        with patch.object(run_b_once, "fit_results_frame", return_value=frame):
            returned_frame, failures = run_b_once.validate_reliability_matrix(
                results,
                ["p1", "p2"],
                {},
            )

        pd.testing.assert_frame_equal(returned_frame, frame)
        self.assertEqual(failures, 8)

    def test_atomic_json_writer_produces_complete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            payload = {"status": "in_progress", "fingerprint": {"x": 1}}
            run_b_once.atomic_write_json(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()