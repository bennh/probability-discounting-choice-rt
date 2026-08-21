import importlib.util
import copy
from pathlib import Path
import unittest


YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(
    YAML_AVAILABLE,
    "PyYAML is unavailable in the current validation runtime",
)
class ConfigTests(unittest.TestCase):
    def test_candidate_config_loads_and_blocks_formal_run_b(self):
        from pd_project.config import (
            assert_formal_run_b_authorized,
            load_config,
            validate_config,
        )

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "analysis.yaml")

        candidate = copy.deepcopy(config)
        candidate["project"]["freeze_status"] = "candidate"
        candidate["project"]["formal_run_b_enabled"] = False

        validate_config(candidate)

        with self.assertRaises(RuntimeError):
            assert_formal_run_b_authorized(candidate)

        missing_readiness = copy.deepcopy(config)
        missing_readiness["project"].pop("pipeline_readiness")
        with self.assertRaises(ValueError):
            validate_config(missing_readiness)

        quoted_false = copy.deepcopy(config)
        quoted_false["project"]["formal_run_b_enabled"] = "false"
        with self.assertRaises(ValueError):
            validate_config(quoted_false)

    def test_data_contract_validation_fails_closed(self):
        from pd_project.config import load_config, validate_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "analysis.yaml")

        missing_expectation = copy.deepcopy(config)
        missing_expectation["data"]["expected"].pop("odds_mismatch_trials")
        with self.assertRaisesRegex(ValueError, "data.expected"):
            validate_config(missing_expectation)

        duplicate_column = copy.deepcopy(config)
        duplicate_column["data"]["columns"]["r_uncert"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_config(duplicate_column)

        invalid_tolerance = copy.deepcopy(config)
        invalid_tolerance["data"]["transforms"]["odds_assert_atol"] = float("inf")
        with self.assertRaisesRegex(ValueError, "odds_assert_atol"):
            validate_config(invalid_tolerance)

        wrong_rt_unit = copy.deepcopy(config)
        wrong_rt_unit["data"]["transforms"]["rt_divisor"] = 1.0
        with self.assertRaisesRegex(ValueError, "RT divisor"):
            validate_config(wrong_rt_unit)

        reversed_actions = copy.deepcopy(config)
        reversed_actions["data"]["coding"]["certain_action"] = 2
        reversed_actions["data"]["coding"]["uncertain_action"] = 1
        with self.assertRaisesRegex(ValueError, "MATLAB label semantics"):
            validate_config(reversed_actions)

        disabled_odds_recomputation = copy.deepcopy(config)
        disabled_odds_recomputation["data"]["transforms"][
            "recompute_odds_from_probability"
        ] = False
        with self.assertRaisesRegex(ValueError, "recompute_odds_from_probability"):
            validate_config(disabled_odds_recomputation)

        coupled_masks = copy.deepcopy(config)
        coupled_masks["data"]["primary_inclusion"][
            "masks_implemented_independently"
        ] = False
        with self.assertRaisesRegex(ValueError, "primary_inclusion"):
            validate_config(coupled_masks)

        fractional_starts = copy.deepcopy(config)
        fractional_starts["optimization"]["multistarts"] = 1.5
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_config(fractional_starts)

        quoted_optimizer_boolean = copy.deepcopy(config)
        quoted_optimizer_boolean["optimization"]["valid_fit"][
            "require_optimizer_success"
        ] = "true"
        with self.assertRaisesRegex(ValueError, "must remain true"):
            validate_config(quoted_optimizer_boolean)

    def test_map_configuration_validation(self):
        from pd_project.config import load_config, validate_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "analysis.yaml")

        mle_config = copy.deepcopy(config)
        mle_config["optimization"]["primary_estimator"] = "mle"
        validate_config(mle_config)

        map_config = copy.deepcopy(config)
        map_config["optimization"]["primary_estimator"] = "map"
        map_config["map_fallback"]["active"] = True
        validate_config(map_config)

        inactive_map = copy.deepcopy(config)
        inactive_map["optimization"]["primary_estimator"] = "map"
        inactive_map["map_fallback"]["active"] = False
        with self.assertRaisesRegex(ValueError, "MAP estimation requires"):
            validate_config(inactive_map)

        invalid_estimator = copy.deepcopy(config)
        invalid_estimator["optimization"]["primary_estimator"] = "invalid"
        with self.assertRaisesRegex(ValueError, "primary_estimator"):
            validate_config(invalid_estimator)

        invalid_g95 = copy.deepcopy(config)
        invalid_g95["map_fallback"]["weak_priors"]["log_b"]["resolved_g95"]["M1"] = 0.0
        with self.assertRaisesRegex(ValueError, "resolved_g95.M1"):
            validate_config(invalid_g95)


if __name__ == "__main__":
    unittest.main()