import importlib.util
import unittest

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


if __name__ == "__main__":
    unittest.main()
