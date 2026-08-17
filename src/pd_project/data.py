"""Load, encode, validate, and audit Probability Discounting trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIDY_COLUMNS = [
    "participant",
    "run",
    "trial_index",
    "condition",
    "r_cert",
    "r_uncert",
    "probability",
    "odds",
    "raw_action",
    "choice_uncertain",
    "rt_seconds",
    "choice_included",
    "rt_included",
]


def choice_mask(raw_action: Any) -> np.ndarray:
    """Return a mask for valid original actions without recoding them first."""

    raw = np.asarray(raw_action, dtype=float)
    return np.isfinite(raw) & np.isin(raw, (1.0, 2.0))


def rt_mask(raw_action: Any, rt_seconds: Any) -> np.ndarray:
    """Return the independently defined RT-likelihood inclusion mask."""

    raw = np.asarray(raw_action, dtype=float)
    rt = np.asarray(rt_seconds, dtype=float)
    if raw.shape != rt.shape:
        raise ValueError("raw_action and rt_seconds must have matching shapes.")
    return choice_mask(raw) & np.isfinite(rt) & (rt > 0.0)


def encode_choice_uncertain(raw_action: Any) -> np.ndarray:
    """Encode action 1 as 0 and action 2 as 1; all other values remain missing."""

    raw = np.asarray(raw_action, dtype=float)
    encoded = np.full(raw.shape, np.nan, dtype=float)
    encoded[raw == 1.0] = 0.0
    encoded[raw == 2.0] = 1.0
    return encoded


def probability_percent_to_unit(probability_percent: Any) -> np.ndarray:
    """Convert explicitly percent-coded probabilities to the open unit interval."""

    values = np.asarray(probability_percent, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= 100.0)):
        raise ValueError("Probability percentages must be finite and between 0 and 100.")
    return values / 100.0


def odds_against(probability_unit: Any) -> np.ndarray:
    """Compute probability-discounting odds against, (1-p)/p."""

    probability = np.asarray(probability_unit, dtype=float)
    if np.any(~np.isfinite(probability)) or np.any(
        (probability <= 0.0) | (probability >= 1.0)
    ):
        raise ValueError("Unit probabilities must be finite and between 0 and 1.")
    return (1.0 - probability) / probability


def compute_s0(run_a_r_cert: Any) -> float:
    """Compute the fixed amount scale from run A only."""

    rewards = np.asarray(run_a_r_cert, dtype=float)
    finite = np.abs(rewards[np.isfinite(rewards)])
    if finite.size == 0:
        raise ValueError("Cannot compute s0 from an empty run-A amount vector.")
    scale = float(np.median(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The run-A amount scale must be finite and positive.")
    return scale


def prepare_trial_matrix(
    matrix: Any,
    *,
    participant: str,
    run: str,
    columns: dict[str, int],
    coding: dict[str, int],
    odds_assert_atol: float,
) -> pd.DataFrame:
    """Convert one MATLAB trial matrix into the canonical tidy schema."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Each run must be a two-dimensional trial matrix.")
    required_width = max(int(index) for index in columns.values()) + 1
    if values.shape[1] < required_width:
        raise ValueError(
            f"Trial matrix has {values.shape[1]} columns; expected at least {required_width}."
        )

    def column(name: str) -> np.ndarray:
        return values[:, int(columns[name])].astype(float, copy=True)

    raw_action = column("raw_action")
    rt_seconds = column("rt_seconds")
    probability = probability_percent_to_unit(column("probability_percent"))
    odds = odds_against(probability)
    source_odds = column("odds_source")
    finite_source = np.isfinite(source_odds)
    if np.any(finite_source) and not np.allclose(
        source_odds[finite_source], odds[finite_source], atol=odds_assert_atol, rtol=0.0
    ):
        max_error = float(np.max(np.abs(source_odds[finite_source] - odds[finite_source])))
        raise ValueError(f"Source odds disagree with recomputed odds (max error={max_error}).")

    condition_code = column("condition_code")
    reward_code = float(coding["reward_condition"])
    loss_code = float(coding["loss_condition"])
    if np.any(~np.isin(condition_code, (reward_code, loss_code))):
        invalid = np.unique(condition_code[~np.isin(condition_code, (reward_code, loss_code))])
        raise ValueError(f"Unknown condition codes: {invalid.tolist()}")
    condition = np.where(condition_code == reward_code, "R", "L")

    frame = pd.DataFrame(
        {
            "participant": str(participant),
            "run": str(run).upper(),
            "trial_index": np.arange(1, values.shape[0] + 1, dtype=int),
            "condition": condition,
            "r_cert": column("r_cert"),
            "r_uncert": column("r_uncert"),
            "probability": probability,
            "odds": odds,
            "raw_action": raw_action,
            "choice_uncertain": encode_choice_uncertain(raw_action),
            "rt_seconds": rt_seconds,
            "choice_included": choice_mask(raw_action),
            "rt_included": rt_mask(raw_action, rt_seconds),
        }
    )
    return frame.loc[:, TIDY_COLUMNS]


def load_participant_mat(path: str | Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """Load one participant file containing run A and run B MATLAB matrices."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("SciPy is required to read MATLAB files.") from exc

    mat_path = Path(path)
    payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = data_config["matlab_keys"]
    missing_keys = [
        key for key in (keys["run_a"], keys["run_b"], keys["labels"]) if key not in payload
    ]
    if missing_keys:
        raise KeyError(f"{mat_path.name} is missing MATLAB keys: {missing_keys}")
    labels = np.atleast_1d(payload[keys["labels"]]).ravel()
    required_width = max(int(index) for index in data_config["columns"].values()) + 1
    if labels.size < required_width:
        raise ValueError(
            f"{mat_path.name} has {labels.size} data labels; expected at least {required_width}."
        )

    common = {
        "participant": mat_path.stem,
        "columns": data_config["columns"],
        "coding": data_config["coding"],
        "odds_assert_atol": float(data_config["transforms"]["odds_assert_atol"]),
    }
    run_a = prepare_trial_matrix(payload[keys["run_a"]], run="A", **common)
    run_b = prepare_trial_matrix(payload[keys["run_b"]], run="B", **common)
    return pd.concat((run_a, run_b), ignore_index=True)


def load_data_directory(
    raw_directory: str | Path, data_config: dict[str, Any]
) -> pd.DataFrame:
    """Load every participant file in deterministic filename order."""

    raw_path = Path(raw_directory)
    files = sorted(raw_path.glob(data_config.get("raw_glob", "*.mat")))
    if not files:
        raise FileNotFoundError(f"No MATLAB files found in {raw_path}.")
    frames = [load_participant_mat(path, data_config) for path in files]
    return pd.concat(frames, ignore_index=True)


def audit_trials(trials: pd.DataFrame, data_config: dict[str, Any]) -> dict[str, Any]:
    """Return observed counts and explicit deviations from frozen expectations."""

    observed = {
        "participants": int(trials["participant"].nunique()),
        "total_trials": int(len(trials)),
        "missing_choice_trials": int((~trials["choice_included"]).sum()),
        "valid_choice_trials": int(trials["choice_included"].sum()),
        "valid_rt_trials": int(trials["rt_included"].sum()),
        "run_counts": {
            str(key): int(value) for key, value in trials["run"].value_counts().sort_index().items()
        },
        "condition_counts": {
            str(key): int(value)
            for key, value in trials["condition"].value_counts().sort_index().items()
        },
    }
    expected = data_config.get("expected", {})
    deviations: dict[str, Any] = {}
    for key in (
        "participants",
        "total_trials",
        "missing_choice_trials",
        "valid_choice_trials",
        "valid_rt_trials",
    ):
        if key in expected and int(expected[key]) != observed[key]:
            deviations[key] = {"expected": int(expected[key]), "observed": observed[key]}
    if "trials_per_participant_run" in expected:
        expected_per_run = int(expected["trials_per_participant_run"])
        participant_run_counts = (
            trials.groupby(["participant", "run"], sort=True).size().rename("observed")
        )
        invalid_counts = participant_run_counts.loc[
            participant_run_counts != expected_per_run
        ]
        if not invalid_counts.empty:
            deviations["trials_per_participant_run"] = {
                "expected": expected_per_run,
                "violations": [
                    {
                        "participant": str(participant),
                        "run": str(run),
                        "observed": int(count),
                    }
                    for (participant, run), count in invalid_counts.items()
                ],
            }
    if "trials_per_condition_per_participant_run" in expected:
        expected_per_condition = int(expected["trials_per_condition_per_participant_run"])
        condition_counts = trials.groupby(
            ["participant", "run", "condition"], sort=True
        ).size()
        invalid_condition_counts = condition_counts.loc[
            condition_counts != expected_per_condition
        ]
        expected_cells = int(observed["participants"]) * 2 * 2
        if len(condition_counts) != expected_cells or not invalid_condition_counts.empty:
            deviations["trials_per_condition_per_participant_run"] = {
                "expected": expected_per_condition,
                "observed_cells": int(len(condition_counts)),
                "expected_cells": expected_cells,
                "violations": [
                    {
                        "participant": str(participant),
                        "run": str(run),
                        "condition": str(condition),
                        "observed": int(count),
                    }
                    for (participant, run, condition), count in invalid_condition_counts.items()
                ],
            }
    if "raw_action_values" in expected and "raw_action" in trials:
        observed_actions = sorted(
            float(value) for value in trials["raw_action"].dropna().unique()
        )
        allowed_actions = sorted(float(value) for value in expected["raw_action_values"])
        observed["raw_action_values"] = observed_actions
        if not set(observed_actions).issubset(set(allowed_actions)):
            deviations["raw_action_values"] = {
                "allowed": allowed_actions,
                "observed": observed_actions,
            }
    if "missing_action_code_trials" in expected and "raw_action" in trials:
        missing_code_count = int((trials["raw_action"] == 0.0).sum())
        observed["missing_action_code_trials"] = missing_code_count
        if missing_code_count != int(expected["missing_action_code_trials"]):
            deviations["missing_action_code_trials"] = {
                "expected": int(expected["missing_action_code_trials"]),
                "observed": missing_code_count,
            }
    if "nan_raw_action_trials" in expected and "raw_action" in trials:
        nan_action_count = int(trials["raw_action"].isna().sum())
        observed["nan_raw_action_trials"] = nan_action_count
        if nan_action_count != int(expected["nan_raw_action_trials"]):
            deviations["nan_raw_action_trials"] = {
                "expected": int(expected["nan_raw_action_trials"]),
                "observed": nan_action_count,
            }
    if "nonfinite_amount_values" in expected:
        amount_values = trials[["r_cert", "r_uncert"]].to_numpy(dtype=float)
        nonfinite_amount_count = int((~np.isfinite(amount_values)).sum())
        observed["nonfinite_amount_values"] = nonfinite_amount_count
        if nonfinite_amount_count != int(expected["nonfinite_amount_values"]):
            deviations["nonfinite_amount_values"] = {
                "expected": int(expected["nonfinite_amount_values"]),
                "observed": nonfinite_amount_count,
            }
    if "probability_levels_unit" in expected and "probability" in trials:
        observed_probabilities = np.sort(
            trials["probability"].dropna().to_numpy(dtype=float)
        )
        observed_levels = np.unique(observed_probabilities)
        expected_levels = np.asarray(expected["probability_levels_unit"], dtype=float)
        observed["probability_levels_unit"] = observed_levels.tolist()
        if observed_levels.shape != expected_levels.shape or not np.allclose(
            observed_levels, expected_levels, atol=1.0e-12, rtol=0.0
        ):
            deviations["probability_levels_unit"] = {
                "expected": expected_levels.tolist(),
                "observed": observed_levels.tolist(),
            }
    if "choice_rt_mask_mismatch_trials" in expected:
        mismatch = int(
            (trials["choice_included"].astype(bool) != trials["rt_included"].astype(bool)).sum()
        )
        observed["choice_rt_mask_mismatch_trials"] = mismatch
        if mismatch != int(expected["choice_rt_mask_mismatch_trials"]):
            deviations["choice_rt_mask_mismatch_trials"] = {
                "expected": int(expected["choice_rt_mask_mismatch_trials"]),
                "observed": mismatch,
            }
    if expected.get("reward_amounts_positive") is True:
        reward = trials.loc[trials["condition"] == "R", ["r_cert", "r_uncert"]]
        violations = int((reward <= 0.0).to_numpy().sum())
        observed["reward_amount_sign_violations"] = violations
        if violations:
            deviations["reward_amounts_positive"] = {"violations": violations}
    if expected.get("loss_amounts_negative") is True:
        loss = trials.loc[trials["condition"] == "L", ["r_cert", "r_uncert"]]
        violations = int((loss >= 0.0).to_numpy().sum())
        observed["loss_amount_sign_violations"] = violations
        if violations:
            deviations["loss_amounts_negative"] = {"violations": violations}
    observed["deviations"] = deviations
    observed["passed_expected_counts"] = not deviations
    return observed
