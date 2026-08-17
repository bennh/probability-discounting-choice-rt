import importlib.util
import copy
from pathlib import Path
import unittest


YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(YAML_AVAILABLE, "PyYAML is unavailable in the current validation runtime")
class ConfigTests(unittest.TestCase):
    def test_candidate_config_loads_and_blocks_formal_run_b(self):
        from pd_project.config import assert_formal_run_b_authorized, load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "analysis.yaml")
        self.assertEqual(config["project"]["freeze_status"], "candidate")
        with self.assertRaises(RuntimeError):
            assert_formal_run_b_authorized(config)

        missing_readiness = copy.deepcopy(config)
        missing_readiness["project"].pop("pipeline_readiness")
        with self.assertRaises(ValueError):
            from pd_project.config import validate_config

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


if __name__ == "__main__":
    unittest.main()
