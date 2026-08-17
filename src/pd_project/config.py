"""Configuration loading, validation, hashing, and run-B safeguards."""

from __future__ import annotations

import hashlib
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

    n_starts = int(config["optimization"].get("multistarts", 0))
    if n_starts < 1:
        raise ValueError("optimization.multistarts must be at least one.")

    bounds = config["optimization"].get("bounds", {})
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
