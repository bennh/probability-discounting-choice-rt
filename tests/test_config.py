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


if __name__ == "__main__":
    unittest.main()
