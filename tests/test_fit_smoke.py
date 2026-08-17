import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is unavailable in the current validation runtime")
class FitSmokeTests(unittest.TestCase):
    def test_one_start_full_fit_returns_finite_objective(self):
        from pd_project.data import odds_against
        from pd_project.fit import fit_participant

        n_trials = 20
        condition = np.where(np.arange(n_trials) % 2 == 0, "R", "L")
        sign = np.where(condition == "R", 1.0, -1.0)
        probability = np.tile([0.25, 0.50, 0.75, 0.90, 0.10], 4)
        trials = pd.DataFrame(
            {
                "participant": "test",
                "condition": condition,
                "r_cert": sign * 10.0,
                "r_uncert": sign * np.linspace(12.0, 24.0, n_trials),
                "odds": odds_against(probability),
                "choice_uncertain": np.arange(n_trials) % 2,
                "rt_seconds": np.linspace(1.0, 2.0, n_trials),
                "choice_included": True,
                "rt_included": True,
            }
        )
        config = {
            "random": {"master_seed": 7},
            "data": {"transforms": {"amount_scale": 10.0}},
            "optimization": {
                "method": "L-BFGS-B",
                "multistarts": 1,
                "max_iterations": 100,
                "ftol": 1e-8,
                "gtol": 1e-5,
                "valid_fit": {"require_optimizer_success": False, "boundary_near_fraction": 0.01},
                "bounds": {
                    "log_k_R": [-3.0, 3.0],
                    "log_k_L": [-3.0, 3.0],
                    "log_beta": [-3.0, 3.0],
                    "alpha": [-2.0, 2.0],
                    "delta_L": [-1.0, 1.0],
                    "log_sigma": [-3.0, 1.0],
                    "log_b": {"M1": [-5.0, 2.0], "M2": [-7.0, 1.0], "M3": [-5.0, 2.0]},
                },
            },
        }
        result = fit_participant(trials, "M1", config, seed=17)
        self.assertTrue(np.isfinite(result.objective))
        self.assertLess(result.objective, 1.0e90)
        self.assertTrue(result.success)

        malformed_mask = trials.copy()
        malformed_mask["choice_included"] = pd.Series(
            ["False"] * len(trials), index=trials.index, dtype=object
        )
        with self.assertRaisesRegex(ValueError, "boolean dtype"):
            fit_participant(malformed_mask, "M1", config, seed=17)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            fit_participant(trials, "M1", config, seed=17, multistarts=1.5)

        with self.assertRaisesRegex(ValueError, "single choice-only baseline"):
            fit_participant(trials, "M2", config, seed=17, choice_only=True)

    def test_penalized_starts_return_explicit_failed_fit(self):
        from pd_project.data import odds_against
        from pd_project.fit import INVALID_OBJECTIVE, fit_participant

        trials = pd.DataFrame(
            {
                "participant": ["test", "test"],
                "condition": ["R", "L"],
                "r_cert": [10.0, -10.0],
                "r_uncert": [20.0, -20.0],
                "odds": odds_against([0.5, 0.5]),
                "choice_uncertain": [1.0, 0.0],
                "rt_seconds": [1.0, 1.0],
                "choice_included": [True, True],
                "rt_included": [True, True],
            }
        )
        config = {
            "data": {"transforms": {"amount_scale": 10.0}},
            "optimization": {
                "method": "L-BFGS-B",
                "multistarts": 1,
                "max_iterations": 10,
                "ftol": 1.0e-8,
                "gtol": 1.0e-5,
                "valid_fit": {
                    "require_optimizer_success": True,
                    "require_finite_objective": True,
                    "boundary_near_fraction": 0.01,
                    "projected_gradient_inf_max": None,
                },
                "bounds": {
                    "log_k_R": [-3.0, 3.0],
                    "log_k_L": [-3.0, 3.0],
                    "log_beta": [-3.0, 3.0],
                    "alpha": [-2.0, 2.0],
                    "delta_L": [-1.0, 1.0],
                    "log_sigma": [-3.0, 1.0],
                    "log_b": {"M1": [-5.0, 2.0]},
                },
            },
        }
        invalid_result = SimpleNamespace(
            fun=INVALID_OBJECTIVE,
            x=np.zeros(7),
            success=True,
            jac=np.zeros(7),
            message="penalty plateau",
            nit=0,
        )
        with patch("pd_project.fit.minimize", return_value=invalid_result):
            result = fit_participant(trials, "M1", config, seed=17)
        self.assertFalse(result.success)
        self.assertEqual(result.best_start_index, -1)
        self.assertTrue(all(np.isnan(value) for value in result.estimates.values()))

    def test_seed_framing_is_unambiguous(self):
        from pd_project.fit import deterministic_seed

        self.assertNotEqual(
            deterministic_seed(7, "a|b", "c"),
            deterministic_seed(7, "a", "b|c"),
        )


if __name__ == "__main__":
    unittest.main()
