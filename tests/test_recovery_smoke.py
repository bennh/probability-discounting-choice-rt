import unittest

import numpy as np

from pd_project.data import odds_against
from pd_project.likelihood import joint_log_likelihood
from pd_project.rt_models import rt_location, rt_predictor
from pd_project.recovery import (
    latin_hypercube_parameters,
    model_recovery_confusion,
    recovery_metrics,
)
import pandas as pd
from pd_project.valuation_choice import choice_logits, subjective_values


class RecoveryKernelSmokeTests(unittest.TestCase):
    def test_recovery_metrics_require_declared_range_width(self):
        with self.assertRaises(ValueError):
            recovery_metrics([0.0, 1.0], [0.1, 0.9])
        metrics = recovery_metrics(
            [0.0, 1.0], [0.1, 0.9], parameter_range=2.0
        )
        self.assertAlmostEqual(metrics["rmse"], 0.1)
        self.assertAlmostEqual(metrics["nrmse"], 0.05)

    def test_model_recovery_aggregates_conditions_before_selecting(self):
        rows = []
        scores = {
            "M1": {"M1": [-1.0, -10.0], "M2": [-2.0, -2.0], "M3": [-3.0, -3.0]},
            "M2": {"M1": [-3.0, -3.0], "M2": [-1.0, -1.0], "M3": [-2.0, -2.0]},
            "M3": {"M1": [-3.0, -3.0], "M2": [-2.0, -2.0], "M3": [-1.0, -1.0]},
        }
        for generating, fitted_scores in scores.items():
            for fitted, condition_scores in fitted_scores.items():
                for condition, score in zip(("R", "L"), condition_scores):
                    rows.append(
                        {
                            "generating_model": generating,
                            "replicate": 0,
                            "fitted_model": fitted,
                            "condition": condition,
                            "run_b_rt_mlpd": score,
                        }
                    )
        confusion = model_recovery_confusion(pd.DataFrame(rows))
        self.assertEqual(confusion.loc["M1", "M2"], 1.0)
        self.assertEqual(confusion.loc["M2", "M2"], 1.0)
        self.assertEqual(confusion.loc["M3", "M3"], 1.0)

    def test_model_recovery_rejects_missing_candidate(self):
        incomplete = pd.DataFrame(
            {
                "generating_model": ["M1", "M1"],
                "replicate": [0, 0],
                "fitted_model": ["M1", "M2"],
                "run_b_rt_mlpd": [-1.0, -2.0],
            }
        )
        with self.assertRaises(ValueError):
            model_recovery_confusion(incomplete)
    def test_latin_hypercube_covers_each_stratum_once(self):
        samples = latin_hypercube_parameters(
            {"alpha": [0.0, 1.0], "log_sigma": [-2.0, 0.0]},
            n_samples=8,
            rng=np.random.default_rng(123),
            log_b_range=(-3.0, 1.0),
        )
        self.assertEqual(len(samples), 8)
        for name, lower, upper in (
            ("alpha", 0.0, 1.0),
            ("log_sigma", -2.0, 0.0),
            ("log_b", -3.0, 1.0),
        ):
            values = np.array([sample[name] for sample in samples])
            self.assertTrue(np.all((values >= lower) & (values <= upper)))
            strata = np.floor((values - lower) / (upper - lower) * 8).astype(int)
            np.testing.assert_array_equal(np.sort(strata), np.arange(8))

    def test_all_models_produce_finite_joint_objective(self):
        condition = np.array(["R", "L", "R", "L"])
        values = subjective_values(
            [10.0, -10.0, 10.0, -10.0],
            [20.0, -20.0, 15.0, -15.0],
            odds_against([0.25, 0.25, 0.75, 0.75]),
            [1.0, 2.0, 1.0, 2.0],
            s0=10.0,
        )
        logits = choice_logits(2.0, values.delta_v)
        choices = np.array([0.0, 1.0, 1.0, 0.0])
        rt = np.array([1.2, 2.1, np.nan, 1.8])
        choice_mask = np.ones(4, dtype=bool)
        rt_mask = np.array([True, True, False, True])
        for model in ("M1", "M2", "M3"):
            predictor = rt_predictor(model, values.v_cert, values.v_uncert, values.delta_v)
            mu = rt_location(0.6, 0.2, 0.3, condition, predictor)
            objective = joint_log_likelihood(
                choices,
                rt,
                logits,
                mu,
                0.4,
                choice_mask=choice_mask,
                rt_mask=rt_mask,
            )
            self.assertTrue(np.isfinite(objective), model)


if __name__ == "__main__":
    unittest.main()
