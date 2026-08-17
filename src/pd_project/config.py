"""Configuration loading, validation, hashing, and run-B safeguards."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml


VALID_FREEZE_STATES = {"candidate", "frozen"}
PRIMARY_MODELS = {"M1", "M2", "M3"}
REQUIRED_READINESS_FLAGS = {
    "formal_parameter_recovery_implemented",
    "formal_model_recovery_implemented",
    "report_statistics_implemented",
    "run_b_one_shot_smoke_tested",
    "notebook_end_to_end_implemented",
    "formal_artifact_archival_approved",
    "remote_run_reservation_approved",
}
SOURCE_COLUMN_NAMES = {
    "r_cert",
    "r_uncert",
    "probability_percent",
    "raw_action",
    "p_cert",
    "condition_code",
    "rt_seconds",
    "odds_source",
}
REQUIRED_DATA_EXPECTATIONS = {
    "participants",
    "run_values",
    "condition_values",
    "trial_index_start",
    "trials_per_participant_run",
    "trials_per_condition_per_participant_run",
    "trials_per_probability_per_condition_per_participant_run",
    "total_trials",
    "missing_choice_trials",
    "valid_choice_trials",
    "valid_rt_trials",
    "raw_action_values",
    "missing_action_code_trials",
    "nan_raw_action_trials",
    "nonfinite_amount_values",
    "nonfinite_probability_values",
    "nonfinite_odds_values",
    "probability_levels_unit",
    "odds_mismatch_trials",
    "choice_encoding_mismatch_trials",
    "choice_mask_definition_mismatch_trials",
    "rt_mask_definition_mismatch_trials",
    "choice_rt_mask_mismatch_trials",
    "missing_action_with_observed_rt_trials",
    "reward_amounts_positive",
    "loss_amounts_negative",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the analysis YAML file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    validate_config(config)
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(config_path.parent.parent)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early when a required analysis contract is missing."""

    required = {
        "project",
        "random",
        "data",
        "valuation",
        "choice",
        "rt",
        "models",
        "optimization",
        "map_fallback",
        "recovery",
        "model_recovery",
        "reliability",
        "prediction",
        "bootstrap",
        "comparison_families",
        "support_shift",
        "run_b_guard",
        "outputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    state = config["project"].get("freeze_status")
    if state not in VALID_FREEZE_STATES:
        raise ValueError(
            f"project.freeze_status must be one of {sorted(VALID_FREEZE_STATES)}."
        )
    readiness = config["project"].get("pipeline_readiness")
    if not isinstance(readiness, dict):
        raise ValueError("project.pipeline_readiness must be a mapping.")
    missing_readiness = sorted(REQUIRED_READINESS_FLAGS - set(readiness))
    if missing_readiness:
        raise ValueError(
            f"Missing project.pipeline_readiness flags: {missing_readiness}"
        )
    non_boolean_readiness = sorted(
        name for name in REQUIRED_READINESS_FLAGS if not isinstance(readiness[name], bool)
    )
    if non_boolean_readiness:
        raise ValueError(
            "Pipeline readiness values must be YAML booleans for: "
            f"{non_boolean_readiness}"
        )
    if not isinstance(config["project"].get("formal_run_b_enabled"), bool):
        raise ValueError("project.formal_run_b_enabled must be a YAML boolean.")

    _validate_data_contract(config["data"])

    for name, specification in config["models"].items():
        if not isinstance(specification.get("enabled"), bool):
            raise ValueError(f"models.{name}.enabled must be a YAML boolean.")
    enabled = {
        name.upper()
        for name, specification in config["models"].items()
        if specification.get("enabled") is True
    }
    if not PRIMARY_MODELS.issubset(enabled):
        raise ValueError("M1, M2, and M3 must all be enabled in the primary analysis.")
    if config["models"].get("M4", {}).get("enabled") is True:
        raise ValueError("M4 is not implemented in the current public model interface.")
    map_active = config["map_fallback"].get("active")
    if not isinstance(map_active, bool):
        raise ValueError("map_fallback.active must be a YAML boolean.")
    if map_active:
        raise ValueError(
            "MAP fallback is not implemented in the current objective; keep it inactive."
        )

    outputs = config["outputs"]
    required_outputs = {"manifest", "run_a_completion_receipt", "recovery_directory"}
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError(f"outputs must define {sorted(required_outputs)}.")
    for name in required_outputs:
        if not isinstance(outputs[name], str) or not outputs[name].strip():
            raise ValueError(f"outputs.{name} must be a non-empty path string.")

    optimization = config["optimization"]
    if optimization.get("method") != "L-BFGS-B":
        raise ValueError("optimization.method must remain L-BFGS-B.")
    n_starts = optimization.get("multistarts")
    if isinstance(n_starts, bool) or not isinstance(n_starts, int) or n_starts < 1:
        raise ValueError("optimization.multistarts must be a positive integer.")
    max_iterations = optimization.get("max_iterations")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("optimization.max_iterations must be a positive integer.")
    for name in ("ftol", "gtol"):
        value = float(optimization.get(name, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"optimization.{name} must be finite and positive.")
    valid_fit = optimization.get("valid_fit")
    if not isinstance(valid_fit, dict):
        raise ValueError("optimization.valid_fit must be a mapping.")
    for name in ("require_optimizer_success", "require_finite_objective"):
        if valid_fit.get(name) is not True:
            raise ValueError(f"optimization.valid_fit.{name} must remain true.")
    boundary_fraction = float(valid_fit.get("boundary_near_fraction", float("nan")))
    if not math.isfinite(boundary_fraction) or not 0.0 <= boundary_fraction <= 0.5:
        raise ValueError(
            "optimization.valid_fit.boundary_near_fraction must be between 0 and 0.5."
        )
    projected_threshold = valid_fit.get("projected_gradient_inf_max")
    if projected_threshold is not None:
        projected_threshold = float(projected_threshold)
        if not math.isfinite(projected_threshold) or projected_threshold < 0.0:
            raise ValueError(
                "optimization.valid_fit.projected_gradient_inf_max must be null "
                "or finite and non-negative."
            )

    master_seed = config["random"].get("master_seed")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("random.master_seed must be an integer.")

    bounds = optimization.get("bounds", {})
    for name in (
        "log_k_R",
        "log_k_L",
        "log_beta",
        "alpha",
        "delta_L",
        "log_sigma",
    ):
        _validate_bound(name, bounds.get(name))
    for model in PRIMARY_MODELS:
        _validate_bound(f"log_b.{model}", bounds.get("log_b", {}).get(model))


def _validate_bound(name: str, bound: Any) -> None:
    if not isinstance(bound, list) or len(bound) != 2:
        raise ValueError(f"optimization.bounds.{name} must be [lower, upper].")
    lower, upper = (float(bound[0]), float(bound[1]))
    if not lower < upper:
        raise ValueError(f"Invalid bound for {name}: lower must be below upper.")


def _validate_data_contract(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping.")
    required_sections = {
        "matlab_keys",
        "columns",
        "column_labels",
        "expected",
        "coding",
        "transforms",
    }
    missing_sections = sorted(required_sections - set(data))
    if missing_sections:
        raise ValueError(f"Missing data-contract sections: {missing_sections}")

    matlab_keys = data["matlab_keys"]
    if not isinstance(matlab_keys, dict) or not {"run_a", "run_b", "labels"}.issubset(
        matlab_keys
    ):
        raise ValueError("data.matlab_keys must define run_a, run_b, and labels.")

    columns = data["columns"]
    if not isinstance(columns, dict) or set(columns) != SOURCE_COLUMN_NAMES:
        raise ValueError(
            "data.columns must define exactly the frozen eight source columns."
        )
    indices = []
    for name, value in columns.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"data.columns.{name} must be a non-negative integer.")
        indices.append(value)
    if set(indices) != set(range(len(SOURCE_COLUMN_NAMES))):
        raise ValueError("data.columns indices must be unique and contiguous from zero.")

    labels = data["column_labels"]
    if not isinstance(labels, dict) or set(labels) != SOURCE_COLUMN_NAMES:
        raise ValueError(
            "data.column_labels must define exactly the frozen eight source columns."
        )
    empty_labels = sorted(name for name, value in labels.items() if not str(value).strip())
    if empty_labels:
        raise ValueError(f"Empty data.column_labels values: {empty_labels}")

    expected = data["expected"]
    if not isinstance(expected, dict):
        raise ValueError("data.expected must be a mapping.")
    missing_expected = sorted(REQUIRED_DATA_EXPECTATIONS - set(expected))
    unknown_expected = sorted(set(expected) - REQUIRED_DATA_EXPECTATIONS)
    if missing_expected or unknown_expected:
        raise ValueError(
            "Invalid data.expected contract; "
            f"missing={missing_expected}, unknown={unknown_expected}."
        )
    if set(str(value) for value in expected["run_values"]) != {"A", "B"}:
        raise ValueError("data.expected.run_values must be exactly [A, B].")
    if set(str(value) for value in expected["condition_values"]) != {"R", "L"}:
        raise ValueError("data.expected.condition_values must be exactly [R, L].")
    if expected["reward_amounts_positive"] is not True:
        raise ValueError("data.expected.reward_amounts_positive must be true.")
    if expected["loss_amounts_negative"] is not True:
        raise ValueError("data.expected.loss_amounts_negative must be true.")

    nonnegative_counts = REQUIRED_DATA_EXPECTATIONS - {
        "run_values",
        "condition_values",
        "raw_action_values",
        "probability_levels_unit",
        "reward_amounts_positive",
        "loss_amounts_negative",
    }
    for name in nonnegative_counts:
        value = expected[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"data.expected.{name} must be a non-negative integer.")
    if expected["trial_index_start"] != 1:
        raise ValueError("data.expected.trial_index_start must be 1.")

    coding = data["coding"]
    required_coding = {
        "missing_action",
        "certain_action",
        "uncertain_action",
        "reward_condition",
        "loss_condition",
    }
    if not isinstance(coding, dict) or not required_coding.issubset(coding):
        raise ValueError(f"data.coding must define {sorted(required_coding)}.")
    numeric_codes = {}
    for name in required_coding:
        try:
            numeric_codes[name] = float(coding[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"data.coding.{name} must be numeric.") from exc
        if not math.isfinite(numeric_codes[name]):
            raise ValueError(f"data.coding.{name} must be finite.")
    action_codes = {
        numeric_codes["missing_action"],
        numeric_codes["certain_action"],
        numeric_codes["uncertain_action"],
    }
    if len(action_codes) != 3:
        raise ValueError("The three action codes must be distinct.")
    if numeric_codes["reward_condition"] == numeric_codes["loss_condition"]:
        raise ValueError("Reward and loss condition codes must be distinct.")
    frozen_codes = {
        "missing_action": 0.0,
        "certain_action": 1.0,
        "uncertain_action": 2.0,
        "reward_condition": 1.0,
        "loss_condition": 2.0,
    }
    if numeric_codes != frozen_codes:
        raise ValueError(
            "data.coding must match the frozen MATLAB label semantics: "
            "missing/certain/uncertain=0/1/2 and reward/loss=1/2."
        )
    if {float(value) for value in expected["raw_action_values"]} != action_codes:
        raise ValueError("data.expected.raw_action_values must match data.coding action codes.")

    probability_levels = [float(value) for value in expected["probability_levels_unit"]]
    if (
        not probability_levels
        or len(probability_levels) != len(set(probability_levels))
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in probability_levels)
    ):
        raise ValueError(
            "data.expected.probability_levels_unit must contain unique open-unit values."
        )

    transforms = data["transforms"]
    for name in ("probability_divisor", "rt_divisor", "amount_scale"):
        value = float(transforms.get(name, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"data.transforms.{name} must be finite and positive.")
    tolerance = float(transforms.get("odds_assert_atol", float("nan")))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "data.transforms.odds_assert_atol must be finite and non-negative."
        )
    if float(transforms["probability_divisor"]) != 100.0:
        raise ValueError("The frozen probability divisor must be 100.")
    if float(transforms["rt_divisor"]) != 1000.0:
        raise ValueError("The frozen RT divisor must be 1000 milliseconds per second.")
    if tolerance != 1.0e-9:
        raise ValueError("The frozen source-odds absolute tolerance must be 1e-9.")
    frozen_transforms = {
        "amount_scale_assertion": "median_abs_r_cert_run_a",
        "recompute_odds_from_probability": True,
        "use_p_cert_predictor": False,
    }
    for name, frozen_value in frozen_transforms.items():
        if transforms.get(name) != frozen_value:
            raise ValueError(
                f"data.transforms.{name} must be {frozen_value!r} for the frozen pipeline."
            )

    primary_inclusion = data.get("primary_inclusion")
    frozen_primary_inclusion = {
        "choice": "raw_action in {1, 2}",
        "rt": "valid choice AND finite(rt_seconds) AND rt_seconds > 0",
        "masks_implemented_independently": True,
        "keep_all_valid_rt": True,
    }
    if not isinstance(primary_inclusion, dict) or primary_inclusion != frozen_primary_inclusion:
        raise ValueError(
            "data.primary_inclusion must exactly match the frozen choice and RT mask contract."
        )

    rt_sensitivity = data.get("rt_sensitivity")
    if not isinstance(rt_sensitivity, dict) or rt_sensitivity.get("enabled") is not True:
        raise ValueError("data.rt_sensitivity must be an enabled mapping.")
    quantiles = rt_sensitivity.get("quantiles_from_run_a")
    provisional = rt_sensitivity.get("provisional_seconds")
    if (
        not isinstance(quantiles, list)
        or len(quantiles) != 2
        or any(not 0.0 < float(value) < 1.0 for value in quantiles)
        or not float(quantiles[0]) < float(quantiles[1])
    ):
        raise ValueError("data.rt_sensitivity quantiles must be two increasing open-unit values.")
    if (
        not isinstance(provisional, list)
        or len(provisional) != 2
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in provisional)
        or not float(provisional[0]) < float(provisional[1])
    ):
        raise ValueError("data.rt_sensitivity.provisional_seconds must be increasing positive values.")
    provisional_atol = float(
        rt_sensitivity.get("provisional_assert_atol", float("nan"))
    )
    if not math.isfinite(provisional_atol) or provisional_atol <= 0.0:
        raise ValueError(
            "data.rt_sensitivity.provisional_assert_atol must be finite and positive."
        )

    participants = expected["participants"]
    per_run = expected["trials_per_participant_run"]
    per_condition = expected["trials_per_condition_per_participant_run"]
    per_probability = expected[
        "trials_per_probability_per_condition_per_participant_run"
    ]
    if per_run != 2 * per_condition:
        raise ValueError("Expected per-run trials must equal two condition cells.")
    if per_condition != len(probability_levels) * per_probability:
        raise ValueError(
            "Expected per-condition trials must equal probability levels times "
            "trials per probability."
        )
    if expected["total_trials"] != participants * 2 * per_run:
        raise ValueError("Expected total_trials is inconsistent with participants and runs.")
    if expected["valid_choice_trials"] + expected["missing_choice_trials"] != expected[
        "total_trials"
    ]:
        raise ValueError("Expected valid and missing choice counts do not sum to total_trials.")
    if expected["valid_rt_trials"] > expected["valid_choice_trials"]:
        raise ValueError("Expected valid_rt_trials cannot exceed valid_choice_trials.")


def repository_path(config: dict[str, Any], configured_path: str) -> Path:
    """Resolve a path relative to the repository containing the config file."""

    root = Path(config["_repository_root"])
    candidate = (root / configured_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Configured path escapes the repository: {configured_path}")
    return candidate


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_formal_run_b_authorized(config: dict[str, Any]) -> None:
    """Require an explicitly frozen and enabled configuration."""

    if config["project"]["freeze_status"] != "frozen":
        raise RuntimeError("Formal run B is blocked: configuration is not frozen.")
    if config["project"].get("formal_run_b_enabled") is not True:
        raise RuntimeError(
            "Formal run B is blocked: project.formal_run_b_enabled is false."
        )
    readiness = config["project"].get("pipeline_readiness")
    if not isinstance(readiness, dict):
        raise RuntimeError("Formal run B is blocked: pipeline_readiness is missing.")
    missing = sorted(REQUIRED_READINESS_FLAGS - set(readiness))
    incomplete = sorted(
        name for name in REQUIRED_READINESS_FLAGS if readiness.get(name) is not True
    )
    if missing:
        raise RuntimeError(
            f"Formal run B is blocked by missing readiness flags: {missing}"
        )
    if incomplete:
        raise RuntimeError(
            "Formal run B is blocked by incomplete pipeline readiness flags: "
            f"{incomplete}"
        )
    gradient_threshold = config["optimization"]["valid_fit"].get(
        "projected_gradient_inf_max"
    )
    if not isinstance(gradient_threshold, (int, float)) or gradient_threshold <= 0:
        raise RuntimeError(
            "Formal run B is blocked until projected_gradient_inf_max is calibrated "
            "and frozen as a positive number."
        )
