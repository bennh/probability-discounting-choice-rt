"""Load, encode, validate, and audit Probability Discounting trials."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


SOURCE_COLUMNS = (
    "r_cert",
    "r_uncert",
    "probability_percent",
    "raw_action",
    "p_cert",
    "condition_code",
    "rt_seconds",
    "odds_source",
)

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

CANONICAL_RUNS = ("A", "B")
CANONICAL_CONDITIONS = ("R", "L")


def _finite_scalar(name: str, value: Any, *, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}.")
    return result


def _choice_codes(certain_action: Any, uncertain_action: Any) -> tuple[float, float]:
    certain = _finite_scalar("certain_action", certain_action)
    uncertain = _finite_scalar("uncertain_action", uncertain_action)
    if certain == uncertain:
        raise ValueError("certain_action and uncertain_action must be distinct.")
    return certain, uncertain


def _coding_values(coding: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(coding, Mapping):
        raise ValueError("data.coding must be a mapping.")
    required = {
        "missing_action",
        "certain_action",
        "uncertain_action",
        "reward_condition",
        "loss_condition",
    }
    missing = sorted(required - set(coding))
    if missing:
        raise ValueError(f"Missing coding fields: {missing}")
    values = {name: _finite_scalar(f"coding.{name}", coding[name]) for name in required}
    action_values = {
        values["missing_action"],
        values["certain_action"],
        values["uncertain_action"],
    }
    if len(action_values) != 3:
        raise ValueError("Missing, certain, and uncertain action codes must be distinct.")
    if values["reward_condition"] == values["loss_condition"]:
        raise ValueError("Reward and loss condition codes must be distinct.")
    return values


def _validated_column_mapping(columns: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(columns, Mapping):
        raise ValueError("data.columns must be a mapping.")
    missing = sorted(set(SOURCE_COLUMNS) - set(columns))
    extra = sorted(set(columns) - set(SOURCE_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Invalid source-column mapping; missing={missing}, extra={extra}."
        )

    validated: dict[str, int] = {}
    for name in SOURCE_COLUMNS:
        index = columns[name]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"Column index for {name} must be a non-negative integer.")
        index = int(index)
        if index < 0:
            raise ValueError(f"Column index for {name} must be non-negative.")
        validated[name] = index

    indices = list(validated.values())
    if len(indices) != len(set(indices)):
        raise ValueError("Source-column indices must be unique.")
    expected_indices = set(range(len(SOURCE_COLUMNS)))
    if set(indices) != expected_indices:
        raise ValueError(
            "Source-column indices must be a contiguous permutation of "
            f"0..{len(SOURCE_COLUMNS) - 1}."
        )
    return validated


def _normalize_matlab_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _normalize_matlab_label(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}:
            return "".join(_normalize_matlab_label(item) for item in value.ravel()).strip()
    return str(value).strip()


def _validate_matlab_labels(
    labels: Any,
    columns: Mapping[str, int],
    expected_labels: Mapping[str, Any],
    *,
    filename: str,
) -> None:
    if not isinstance(expected_labels, Mapping):
        raise ValueError("data.column_labels must be a mapping.")
    missing = sorted(set(SOURCE_COLUMNS) - set(expected_labels))
    extra = sorted(set(expected_labels) - set(SOURCE_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Invalid column-label contract; missing={missing}, extra={extra}."
        )

    observed = [_normalize_matlab_label(value) for value in np.atleast_1d(labels).ravel()]
    if len(observed) != len(SOURCE_COLUMNS):
        raise ValueError(
            f"{filename} has {len(observed)} data labels; "
            f"expected exactly {len(SOURCE_COLUMNS)}."
        )
    mismatches = []
    for name in SOURCE_COLUMNS:
        index = int(columns[name])
        expected = _normalize_matlab_label(expected_labels[name])
        if observed[index] != expected:
            mismatches.append(
                {
                    "column": name,
                    "index": index,
                    "expected": expected,
                    "observed": observed[index],
                }
            )
    if mismatches:
        raise ValueError(f"{filename} MATLAB data labels do not match: {mismatches}")


def choice_mask(
    raw_action: Any,
    *,
    certain_action: float = 1.0,
    uncertain_action: float = 2.0,
) -> np.ndarray:
    """Return a mask for valid original actions without recoding them first."""

    certain, uncertain = _choice_codes(certain_action, uncertain_action)
    raw = np.asarray(raw_action, dtype=float)
    return np.isfinite(raw) & np.isin(raw, (certain, uncertain))


def rt_mask(
    raw_action: Any,
    rt_seconds: Any,
    *,
    certain_action: float = 1.0,
    uncertain_action: float = 2.0,
) -> np.ndarray:
    """Return the independently derived RT-likelihood inclusion mask."""

    raw = np.asarray(raw_action, dtype=float)
    rt = np.asarray(rt_seconds, dtype=float)
    if raw.shape != rt.shape:
        raise ValueError("raw_action and rt_seconds must have matching shapes.")
    return choice_mask(
        raw,
        certain_action=certain_action,
        uncertain_action=uncertain_action,
    ) & np.isfinite(rt) & (rt > 0.0)


def encode_choice_uncertain(
    raw_action: Any,
    *,
    certain_action: float = 1.0,
    uncertain_action: float = 2.0,
) -> np.ndarray:
    """Encode the certain action as 0 and uncertain action as 1."""

    certain, uncertain = _choice_codes(certain_action, uncertain_action)
    raw = np.asarray(raw_action, dtype=float)
    encoded = np.full(raw.shape, np.nan, dtype=float)
    encoded[raw == certain] = 0.0
    encoded[raw == uncertain] = 1.0
    return encoded


def probability_percent_to_unit(
    probability_percent: Any, *, divisor: float = 100.0
) -> np.ndarray:
    """Convert explicitly percent-coded probabilities to the open unit interval."""

    scale = _finite_scalar("probability divisor", divisor, positive=True)
    values = np.asarray(probability_percent, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= scale)):
        raise ValueError(
            f"Probability values must be finite and strictly between 0 and {scale}."
        )
    return values / scale


def rt_source_to_seconds(rt_source: Any, *, divisor: float = 1000.0) -> np.ndarray:
    """Convert source RT units to seconds while preserving missing values."""

    scale = _finite_scalar("RT divisor", divisor, positive=True)
    return np.asarray(rt_source, dtype=float) / scale


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
    probability_divisor: float = 100.0,
    rt_divisor: float = 1000.0,
) -> pd.DataFrame:
    """Convert one MATLAB trial matrix into the canonical tidy schema."""

    column_indices = _validated_column_mapping(columns)
    coding_values = _coding_values(coding)
    tolerance = _finite_scalar("odds_assert_atol", odds_assert_atol)
    if tolerance < 0.0:
        raise ValueError("odds_assert_atol must be non-negative.")

    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Each run must be a two-dimensional trial matrix.")
    if values.shape[0] == 0:
        raise ValueError("A trial matrix cannot be empty.")
    required_width = len(SOURCE_COLUMNS)
    if values.shape[1] != required_width:
        hint = " The matrix may be transposed." if values.shape[0] == required_width else ""
        raise ValueError(
            f"Trial matrix has shape {values.shape}; expected (*, {required_width}).{hint}"
        )

    participant_id = str(participant).strip()
    if not participant_id:
        raise ValueError("participant must be a non-empty identifier.")
    run_label = str(run).upper()
    if run_label not in CANONICAL_RUNS:
        raise ValueError(f"Run label must be one of {list(CANONICAL_RUNS)}.")

    def column(name: str) -> np.ndarray:
        return values[:, column_indices[name]].astype(float, copy=True)

    raw_action = column("raw_action")
    allowed_actions = (
        coding_values["missing_action"],
        coding_values["certain_action"],
        coding_values["uncertain_action"],
    )
    invalid_action = ~np.isnan(raw_action) & ~np.isin(raw_action, allowed_actions)
    if np.any(invalid_action):
        invalid = np.unique(raw_action[invalid_action])
        raise ValueError(f"Unknown finite action codes: {invalid.tolist()}")

    rt_source = column("rt_seconds")
    if np.any(np.isinf(rt_source)):
        count = int(np.isinf(rt_source).sum())
        raise ValueError(f"Source RT cannot contain infinite values; found {count}.")
    rt_seconds = rt_source_to_seconds(rt_source, divisor=rt_divisor)
    probability = probability_percent_to_unit(
        column("probability_percent"), divisor=probability_divisor
    )
    odds = odds_against(probability)
    source_odds = column("odds_source")
    if np.any(~np.isfinite(source_odds)):
        count = int((~np.isfinite(source_odds)).sum())
        raise ValueError(f"Source odds must be finite; found {count} invalid values.")
    if not np.allclose(source_odds, odds, atol=tolerance, rtol=0.0):
        max_error = float(np.max(np.abs(source_odds - odds)))
        raise ValueError(f"Source odds disagree with recomputed odds (max error={max_error}).")

    condition_code = column("condition_code")
    reward_code = coding_values["reward_condition"]
    loss_code = coding_values["loss_condition"]
    if np.any(~np.isin(condition_code, (reward_code, loss_code))):
        invalid = np.unique(condition_code[~np.isin(condition_code, (reward_code, loss_code))])
        raise ValueError(f"Unknown condition codes: {invalid.tolist()}")
    condition = np.where(condition_code == reward_code, "R", "L")

    certain_action = coding_values["certain_action"]
    uncertain_action = coding_values["uncertain_action"]
    frame = pd.DataFrame(
        {
            "participant": participant_id,
            "run": run_label,
            "trial_index": np.arange(1, values.shape[0] + 1, dtype=int),
            "condition": condition,
            "r_cert": column("r_cert"),
            "r_uncert": column("r_uncert"),
            "probability": probability,
            "odds": odds,
            "raw_action": raw_action,
            "choice_uncertain": encode_choice_uncertain(
                raw_action,
                certain_action=certain_action,
                uncertain_action=uncertain_action,
            ),
            "rt_seconds": rt_seconds,
            "choice_included": choice_mask(
                raw_action,
                certain_action=certain_action,
                uncertain_action=uncertain_action,
            ),
            "rt_included": rt_mask(
                raw_action,
                rt_seconds,
                certain_action=certain_action,
                uncertain_action=uncertain_action,
            ),
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

    columns = _validated_column_mapping(data_config["columns"])
    _validate_matlab_labels(
        payload[keys["labels"]],
        columns,
        data_config["column_labels"],
        filename=mat_path.name,
    )

    transforms = data_config["transforms"]
    common = {
        "participant": mat_path.stem,
        "columns": columns,
        "coding": data_config["coding"],
        "odds_assert_atol": float(transforms["odds_assert_atol"]),
        "probability_divisor": float(transforms["probability_divisor"]),
        "rt_divisor": float(transforms["rt_divisor"]),
    }
    run_a = prepare_trial_matrix(payload[keys["run_a"]], run="A", **common)
    run_b = prepare_trial_matrix(payload[keys["run_b"]], run="B", **common)
    return pd.concat((run_a, run_b), ignore_index=True)


def load_data_files(
    paths: Any, data_config: dict[str, Any]
) -> pd.DataFrame:
    """Load an explicit, deterministic generation of participant MAT files."""

    files = sorted(Path(path) for path in paths)
    if not files:
        raise FileNotFoundError("No MATLAB participant files were supplied.")
    participant_ids = [path.stem for path in files]
    duplicates = sorted(
        identifier for identifier in set(participant_ids) if participant_ids.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate participant filenames found: {duplicates}")
    frames = [load_participant_mat(path, data_config) for path in files]
    return pd.concat(frames, ignore_index=True)


def load_data_directory(
    raw_directory: str | Path, data_config: dict[str, Any]
) -> pd.DataFrame:
    """Load every participant file in deterministic filename order."""

    raw_path = Path(raw_directory)
    files = sorted(raw_path.glob(data_config.get("raw_glob", "*.mat")))
    if not files:
        raise FileNotFoundError(f"No MATLAB files found in {raw_path}.")
    return load_data_files(files, data_config)


def _strict_bool_column(trials: pd.DataFrame, name: str) -> np.ndarray:
    series = trials[name]
    if not is_bool_dtype(series.dtype) or series.isna().any():
        raise ValueError(f"{name} must contain non-missing boolean values only.")
    return series.to_numpy(dtype=bool)


def _numeric_column(trials: pd.DataFrame, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(trials[name], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc


def audit_trials(trials: pd.DataFrame, data_config: dict[str, Any]) -> dict[str, Any]:
    """Audit counts, identities, domains, and all derived trial-level fields."""

    missing_columns = sorted(set(TIDY_COLUMNS) - set(trials.columns))
    if missing_columns:
        raise ValueError(f"Tidy trials are missing required columns: {missing_columns}")
    expected = data_config.get("expected")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("data.expected must be a non-empty mapping.")

    count_deviations: dict[str, Any] = {}
    integrity_deviations: dict[str, Any] = {}
    coding = _coding_values(data_config["coding"])
    transforms = data_config["transforms"]
    odds_tolerance = _finite_scalar("odds_assert_atol", transforms["odds_assert_atol"])
    if odds_tolerance < 0.0:
        raise ValueError("odds_assert_atol must be non-negative.")

    participant_missing = trials["participant"].isna()
    participant_source = trials["participant"].astype("string")
    participant_text = participant_source.str.strip()
    participant_invalid = participant_missing | participant_text.eq("").fillna(True)
    if participant_invalid.any():
        integrity_deviations["invalid_participant_values"] = {
            "observed": int(participant_invalid.sum())
        }
    participant_whitespace = (
        ~participant_missing & participant_source.ne(participant_text).fillna(False)
    )
    if participant_whitespace.any():
        integrity_deviations["participant_whitespace"] = {
            "observed": int(participant_whitespace.sum())
        }
    participants = sorted(participant_text.loc[~participant_invalid].astype(str).unique())

    run_missing = trials["run"].isna()
    run_source = trials["run"].astype("string")
    run_text = run_source.str.strip()
    condition_missing = trials["condition"].isna()
    condition_source = trials["condition"].astype("string")
    condition_text = condition_source.str.strip()
    expected_runs = [str(value) for value in expected["run_values"]]
    expected_conditions = [str(value) for value in expected["condition_values"]]
    observed_runs = sorted(run_text.loc[~run_missing].astype(str).unique())
    observed_conditions = sorted(
        condition_text.loc[~condition_missing].astype(str).unique()
    )
    if run_missing.any() or set(observed_runs) != set(expected_runs):
        integrity_deviations["run_values"] = {
            "expected": expected_runs,
            "observed": observed_runs,
            "missing_values": int(run_missing.sum()),
        }
    if condition_missing.any() or set(observed_conditions) != set(expected_conditions):
        integrity_deviations["condition_values"] = {
            "expected": expected_conditions,
            "observed": observed_conditions,
            "missing_values": int(condition_missing.sum()),
        }
    label_whitespace = (
        (~run_missing & run_source.ne(run_text).fillna(False))
        | (~condition_missing & condition_source.ne(condition_text).fillna(False))
    )
    if label_whitespace.any():
        integrity_deviations["run_condition_whitespace"] = {
            "observed": int(label_whitespace.sum())
        }

    choice_included = _strict_bool_column(trials, "choice_included")
    rt_included = _strict_bool_column(trials, "rt_included")
    observed: dict[str, Any] = {
        "participants": len(participants),
        "total_trials": int(len(trials)),
        "missing_choice_trials": int((~choice_included).sum()),
        "valid_choice_trials": int(choice_included.sum()),
        "valid_rt_trials": int(rt_included.sum()),
        "run_counts": {
            str(key): int(value)
            for key, value in run_text.value_counts(dropna=False).sort_index().items()
        },
        "condition_counts": {
            str(key): int(value)
            for key, value in condition_text.value_counts(dropna=False).sort_index().items()
        },
    }
    for key in (
        "participants",
        "total_trials",
        "missing_choice_trials",
        "valid_choice_trials",
        "valid_rt_trials",
    ):
        if int(expected[key]) != observed[key]:
            count_deviations[key] = {
                "expected": int(expected[key]),
                "observed": observed[key],
            }

    expected_per_run = int(expected["trials_per_participant_run"])
    expected_run_index = pd.MultiIndex.from_product(
        [participants, expected_runs], names=["participant", "run"]
    )
    participant_run_counts = (
        pd.DataFrame(
            {
                "participant": participant_text.astype("string"),
                "run": run_text.astype("string"),
            }
        )
        .groupby(["participant", "run"], sort=True, dropna=False)
        .size()
        .reindex(expected_run_index, fill_value=0)
    )
    invalid_run_counts = participant_run_counts.loc[
        participant_run_counts != expected_per_run
    ]
    if not invalid_run_counts.empty:
        count_deviations["trials_per_participant_run"] = {
            "expected": expected_per_run,
            "violations": [
                {
                    "participant": str(participant),
                    "run": str(run),
                    "observed": int(count),
                }
                for (participant, run), count in invalid_run_counts.items()
            ],
        }

    expected_per_condition = int(expected["trials_per_condition_per_participant_run"])
    expected_condition_index = pd.MultiIndex.from_product(
        [participants, expected_runs, expected_conditions],
        names=["participant", "run", "condition"],
    )
    condition_counts = (
        pd.DataFrame(
            {
                "participant": participant_text.astype("string"),
                "run": run_text.astype("string"),
                "condition": condition_text.astype("string"),
            }
        )
        .groupby(["participant", "run", "condition"], sort=True, dropna=False)
        .size()
        .reindex(expected_condition_index, fill_value=0)
    )
    invalid_condition_counts = condition_counts.loc[
        condition_counts != expected_per_condition
    ]
    if not invalid_condition_counts.empty:
        count_deviations["trials_per_condition_per_participant_run"] = {
            "expected": expected_per_condition,
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

    trial_index = _numeric_column(trials, "trial_index")
    noninteger_trial_index = ~np.isfinite(trial_index) | (trial_index != np.floor(trial_index))
    if np.any(noninteger_trial_index):
        integrity_deviations["invalid_trial_index_values"] = {
            "observed": int(noninteger_trial_index.sum())
        }
    identity_frame = pd.DataFrame(
        {
            "participant": participant_text.astype("string"),
            "run": run_text.astype("string"),
            "trial_index": trial_index,
        }
    )
    valid_identity = ~participant_invalid & ~run_missing & ~noninteger_trial_index
    duplicate_identity = identity_frame.loc[valid_identity].duplicated(
        ["participant", "run", "trial_index"], keep=False
    )
    if duplicate_identity.any():
        integrity_deviations["duplicate_participant_run_trial"] = {
            "observed_rows": int(duplicate_identity.sum())
        }
    start_index = int(expected["trial_index_start"])
    sequence_violations = []
    valid_identity_frame = identity_frame.loc[valid_identity]
    for (participant, run), group in valid_identity_frame.groupby(
        ["participant", "run"], sort=True
    ):
        indices = np.sort(group["trial_index"].to_numpy(dtype=int))
        target = np.arange(start_index, start_index + len(indices), dtype=int)
        if not np.array_equal(indices, target):
            sequence_violations.append(
                {
                    "participant": str(participant),
                    "run": str(run),
                    "observed_min": int(indices.min()) if indices.size else None,
                    "observed_max": int(indices.max()) if indices.size else None,
                    "observed_unique": int(np.unique(indices).size),
                }
            )
    if sequence_violations:
        integrity_deviations["trial_index_sequence"] = {
            "start": start_index,
            "violations": sequence_violations,
        }

    raw_action = _numeric_column(trials, "raw_action")
    infinite_action_count = int(np.isinf(raw_action).sum())
    observed["infinite_raw_action_trials"] = infinite_action_count
    if infinite_action_count:
        integrity_deviations["infinite_raw_action_trials"] = {
            "observed": infinite_action_count
        }
    observed_actions = sorted(float(value) for value in raw_action[np.isfinite(raw_action)])
    observed_action_levels = sorted(set(observed_actions))
    expected_action_levels = sorted(float(value) for value in expected["raw_action_values"])
    observed["raw_action_values"] = observed_action_levels
    if observed_action_levels != expected_action_levels:
        integrity_deviations["raw_action_values"] = {
            "expected": expected_action_levels,
            "observed": observed_action_levels,
        }

    missing_action_count = int((raw_action == coding["missing_action"]).sum())
    nan_action_count = int(np.isnan(raw_action).sum())
    observed["missing_action_code_trials"] = missing_action_count
    observed["nan_raw_action_trials"] = nan_action_count
    for key, value in (
        ("missing_action_code_trials", missing_action_count),
        ("nan_raw_action_trials", nan_action_count),
    ):
        if value != int(expected[key]):
            count_deviations[key] = {"expected": int(expected[key]), "observed": value}

    r_cert = _numeric_column(trials, "r_cert")
    r_uncert = _numeric_column(trials, "r_uncert")
    nonfinite_amount_count = int((~np.isfinite(np.column_stack((r_cert, r_uncert)))).sum())
    observed["nonfinite_amount_values"] = nonfinite_amount_count
    if nonfinite_amount_count != int(expected["nonfinite_amount_values"]):
        integrity_deviations["nonfinite_amount_values"] = {
            "expected": int(expected["nonfinite_amount_values"]),
            "observed": nonfinite_amount_count,
        }

    condition_array = condition_text.fillna("").to_numpy(dtype=str)
    if expected.get("reward_amounts_positive") is True:
        reward_mask = condition_array == "R"
        violations = int(
            ((r_cert[reward_mask] <= 0.0) | (r_uncert[reward_mask] <= 0.0)).sum()
        )
        observed["reward_amount_sign_violations"] = violations
        if violations:
            integrity_deviations["reward_amounts_positive"] = {"violations": violations}
    if expected.get("loss_amounts_negative") is True:
        loss_mask = condition_array == "L"
        violations = int(
            ((r_cert[loss_mask] >= 0.0) | (r_uncert[loss_mask] >= 0.0)).sum()
        )
        observed["loss_amount_sign_violations"] = violations
        if violations:
            integrity_deviations["loss_amounts_negative"] = {"violations": violations}

    probability = _numeric_column(trials, "probability")
    nonfinite_probability = int((~np.isfinite(probability)).sum())
    invalid_probability = int(
        (np.isfinite(probability) & ((probability <= 0.0) | (probability >= 1.0))).sum()
    )
    observed["nonfinite_probability_values"] = nonfinite_probability
    observed["invalid_probability_values"] = invalid_probability
    if nonfinite_probability != int(expected["nonfinite_probability_values"]):
        integrity_deviations["nonfinite_probability_values"] = {
            "expected": int(expected["nonfinite_probability_values"]),
            "observed": nonfinite_probability,
        }
    if invalid_probability:
        integrity_deviations["invalid_probability_values"] = {
            "observed": invalid_probability
        }
    observed_levels = np.unique(probability[np.isfinite(probability)])
    expected_levels = np.asarray(expected["probability_levels_unit"], dtype=float)
    observed["probability_levels_unit"] = observed_levels.tolist()
    if observed_levels.shape != expected_levels.shape or not np.allclose(
        observed_levels, expected_levels, atol=1.0e-12, rtol=0.0
    ):
        integrity_deviations["probability_levels_unit"] = {
            "expected": expected_levels.tolist(),
            "observed": observed_levels.tolist(),
        }
    if "trials_per_probability_per_condition_per_participant_run" in expected:
        expected_per_probability = int(
            expected["trials_per_probability_per_condition_per_participant_run"]
        )
        expected_probability_index = pd.MultiIndex.from_product(
            [participants, expected_runs, expected_conditions, expected_levels.tolist()],
            names=["participant", "run", "condition", "probability"],
        )
        probability_counts = (
            pd.DataFrame(
                {
                    "participant": participant_text.astype("string"),
                    "run": run_text.astype("string"),
                    "condition": condition_text.astype("string"),
                    "probability": probability,
                }
            )
            .groupby(
                ["participant", "run", "condition", "probability"],
                sort=True,
                dropna=False,
            )
            .size()
            .reindex(expected_probability_index, fill_value=0)
        )
        invalid_probability_counts = probability_counts.loc[
            probability_counts != expected_per_probability
        ]
        if not invalid_probability_counts.empty:
            integrity_deviations[
                "trials_per_probability_per_condition_per_participant_run"
            ] = {
                "expected": expected_per_probability,
                "violations": [
                    {
                        "participant": str(participant),
                        "run": str(run),
                        "condition": str(condition),
                        "probability": float(probability_value),
                        "observed": int(count),
                    }
                    for (
                        participant,
                        run,
                        condition,
                        probability_value,
                    ), count in invalid_probability_counts.items()
                ],
            }

    odds = _numeric_column(trials, "odds")
    nonfinite_odds = int((~np.isfinite(odds)).sum())
    observed["nonfinite_odds_values"] = nonfinite_odds
    if nonfinite_odds != int(expected["nonfinite_odds_values"]):
        integrity_deviations["nonfinite_odds_values"] = {
            "expected": int(expected["nonfinite_odds_values"]),
            "observed": nonfinite_odds,
        }
    valid_odds_comparison = (
        np.isfinite(probability)
        & (probability > 0.0)
        & (probability < 1.0)
        & np.isfinite(odds)
    )
    recomputed_odds = np.full(len(trials), np.nan, dtype=float)
    recomputed_odds[valid_odds_comparison] = (
        1.0 - probability[valid_odds_comparison]
    ) / probability[valid_odds_comparison]
    odds_mismatch = valid_odds_comparison & ~np.isclose(
        odds, recomputed_odds, atol=odds_tolerance, rtol=0.0
    )
    odds_mismatch_count = int(odds_mismatch.sum())
    observed["odds_mismatch_trials"] = odds_mismatch_count
    if odds_mismatch_count != int(expected["odds_mismatch_trials"]):
        integrity_deviations["odds_mismatch_trials"] = {
            "expected": int(expected["odds_mismatch_trials"]),
            "observed": odds_mismatch_count,
        }

    certain_action = coding["certain_action"]
    uncertain_action = coding["uncertain_action"]
    encoded_choice = _numeric_column(trials, "choice_uncertain")
    recomputed_choice = encode_choice_uncertain(
        raw_action,
        certain_action=certain_action,
        uncertain_action=uncertain_action,
    )
    choice_equal = (np.isnan(encoded_choice) & np.isnan(recomputed_choice)) | (
        encoded_choice == recomputed_choice
    )
    choice_encoding_mismatch = int((~choice_equal).sum())
    observed["choice_encoding_mismatch_trials"] = choice_encoding_mismatch
    if choice_encoding_mismatch != int(expected["choice_encoding_mismatch_trials"]):
        integrity_deviations["choice_encoding_mismatch_trials"] = {
            "expected": int(expected["choice_encoding_mismatch_trials"]),
            "observed": choice_encoding_mismatch,
        }

    recomputed_choice_mask = choice_mask(
        raw_action,
        certain_action=certain_action,
        uncertain_action=uncertain_action,
    )
    choice_mask_mismatch = int((choice_included != recomputed_choice_mask).sum())
    observed["choice_mask_definition_mismatch_trials"] = choice_mask_mismatch
    if choice_mask_mismatch != int(expected["choice_mask_definition_mismatch_trials"]):
        integrity_deviations["choice_mask_definition_mismatch_trials"] = {
            "expected": int(expected["choice_mask_definition_mismatch_trials"]),
            "observed": choice_mask_mismatch,
        }

    rt_seconds = _numeric_column(trials, "rt_seconds")
    infinite_rt_count = int(np.isinf(rt_seconds).sum())
    observed["infinite_rt_values"] = infinite_rt_count
    if infinite_rt_count:
        integrity_deviations["infinite_rt_values"] = {
            "observed": infinite_rt_count
        }
    recomputed_rt_mask = rt_mask(
        raw_action,
        rt_seconds,
        certain_action=certain_action,
        uncertain_action=uncertain_action,
    )
    rt_mask_mismatch = int((rt_included != recomputed_rt_mask).sum())
    observed["rt_mask_definition_mismatch_trials"] = rt_mask_mismatch
    if rt_mask_mismatch != int(expected["rt_mask_definition_mismatch_trials"]):
        integrity_deviations["rt_mask_definition_mismatch_trials"] = {
            "expected": int(expected["rt_mask_definition_mismatch_trials"]),
            "observed": rt_mask_mismatch,
        }

    rt_sensitivity = data_config.get("rt_sensitivity")
    if isinstance(rt_sensitivity, Mapping) and rt_sensitivity.get("enabled") is True:
        quantiles = np.asarray(rt_sensitivity["quantiles_from_run_a"], dtype=float)
        run_a_valid_rt = (
            run_text.fillna("").to_numpy(dtype=str) == "A"
        ) & rt_included
        if quantiles.shape != (2,) or np.any(~np.isfinite(quantiles)) or np.any(
            (quantiles <= 0.0) | (quantiles >= 1.0)
        ):
            raise ValueError("Run-A RT sensitivity quantiles must contain two open-unit values.")
        if not np.any(run_a_valid_rt):
            integrity_deviations["run_a_rt_sensitivity_seconds"] = {
                "error": "no valid run-A RT values"
            }
        else:
            observed_cutoffs = np.quantile(rt_seconds[run_a_valid_rt], quantiles)
            observed["run_a_rt_sensitivity_seconds"] = observed_cutoffs.tolist()
            provisional = np.asarray(
                rt_sensitivity["provisional_seconds"], dtype=float
            )
            provisional_atol = _finite_scalar(
                "rt_sensitivity.provisional_assert_atol",
                rt_sensitivity["provisional_assert_atol"],
                positive=True,
            )
            if provisional.shape != (2,) or not np.allclose(
                observed_cutoffs,
                provisional,
                atol=provisional_atol,
                rtol=0.0,
            ):
                integrity_deviations["run_a_rt_sensitivity_seconds"] = {
                    "expected_provisional": provisional.tolist(),
                    "observed": observed_cutoffs.tolist(),
                    "atol": provisional_atol,
                }
            resolved = rt_sensitivity.get("resolved_seconds")
            if resolved is not None:
                resolved_values = np.asarray(resolved, dtype=float)
                if resolved_values.shape != (2,) or not np.allclose(
                    observed_cutoffs,
                    resolved_values,
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    integrity_deviations["resolved_run_a_rt_sensitivity_seconds"] = {
                        "expected": resolved_values.tolist(),
                        "observed": observed_cutoffs.tolist(),
                    }

    mask_mismatch = int((choice_included != rt_included).sum())
    observed["choice_rt_mask_mismatch_trials"] = mask_mismatch
    if mask_mismatch != int(expected["choice_rt_mask_mismatch_trials"]):
        count_deviations["choice_rt_mask_mismatch_trials"] = {
            "expected": int(expected["choice_rt_mask_mismatch_trials"]),
            "observed": mask_mismatch,
        }

    missing_action_with_observed_rt = int(
        ((raw_action == coding["missing_action"]) & np.isfinite(rt_seconds)).sum()
    )
    observed["missing_action_with_observed_rt_trials"] = missing_action_with_observed_rt
    if missing_action_with_observed_rt != int(
        expected["missing_action_with_observed_rt_trials"]
    ):
        count_deviations["missing_action_with_observed_rt_trials"] = {
            "expected": int(expected["missing_action_with_observed_rt_trials"]),
            "observed": missing_action_with_observed_rt,
        }

    deviations = {**count_deviations, **integrity_deviations}
    observed["count_deviations"] = count_deviations
    observed["integrity_deviations"] = integrity_deviations
    observed["deviations"] = deviations
    observed["passed_integrity_checks"] = not integrity_deviations
    observed["passed_frozen_contract"] = not deviations
    observed["approved_for_fitting"] = not deviations
    # Backward-compatible alias used by the existing pipeline guards.
    observed["passed_expected_counts"] = not deviations
    return observed
