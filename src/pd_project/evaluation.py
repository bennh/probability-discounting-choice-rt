"""Held-out predictive scores, support checks, and paired resampling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .likelihood import bernoulli_logpmf_from_logits, lognormal_logpdf
from .valuation_choice import choice_probability


def _one_dimensional(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _boolean_mask(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional boolean mask.")
    return array.astype(bool, copy=False)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _finite_statistic(
    statistic: Callable[[Any], float], values: Any, *, context: str
) -> float:
    result = float(statistic(values))
    if not np.isfinite(result):
        raise ValueError(f"Bootstrap statistic is non-finite for {context}.")
    return result


def trial_scores(
    choice_uncertain: Any,
    rt_seconds: Any,
    logits: Any,
    mu: Any,
    sigma: float,
    *,
    choice_included: Any,
    rt_included: Any,
) -> pd.DataFrame:
    """Return trial-level predictive quantities without aggregating participants."""

    y = _one_dimensional("choice_uncertain", choice_uncertain, dtype=float)
    rt = _one_dimensional("rt_seconds", rt_seconds, dtype=float)
    z = _one_dimensional("logits", logits, dtype=float)
    location = _one_dimensional("mu", mu, dtype=float)
    choice_mask = _boolean_mask("choice_included", choice_included)
    rt_mask = _boolean_mask("rt_included", rt_included)
    if len({array.shape for array in (y, rt, z, location, choice_mask, rt_mask)}) != 1:
        raise ValueError("All trial-score inputs must have matching shapes.")
    if np.any(rt_mask & ~choice_mask):
        raise ValueError("Every scored RT trial must also have a valid choice.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("RT sigma must be finite and positive.")

    output = pd.DataFrame(
        {
            "choice_log_score": np.nan,
            "brier_score": np.nan,
            "choice_correct": np.nan,
            "rt_log_score": np.nan,
            "absolute_log_rt_error": np.nan,
            "absolute_rt_error_seconds": np.nan,
        },
        index=np.arange(y.size),
    )
    probability = choice_probability(z[choice_mask])
    output.loc[choice_mask, "choice_log_score"] = bernoulli_logpmf_from_logits(
        y[choice_mask], z[choice_mask]
    )
    output.loc[choice_mask, "brier_score"] = (
        probability - y[choice_mask]
    ) ** 2
    output.loc[choice_mask, "choice_correct"] = (
        (probability >= 0.5) == y[choice_mask]
    ).astype(float)
    output.loc[rt_mask, "rt_log_score"] = lognormal_logpdf(
        rt[rt_mask], location[rt_mask], sigma
    )
    output.loc[rt_mask, "absolute_log_rt_error"] = np.abs(
        np.log(rt[rt_mask]) - location[rt_mask]
    )
    output.loc[rt_mask, "absolute_rt_error_seconds"] = np.abs(
        rt[rt_mask] - np.exp(location[rt_mask])
    )
    return output


def participant_condition_means(
    scored_trials: pd.DataFrame, score_columns: list[str]
) -> pd.DataFrame:
    """Aggregate trials within participant x condition before group summaries."""

    required = {"participant", "condition", *score_columns}
    missing = sorted(required - set(scored_trials.columns))
    if missing:
        raise ValueError(f"Scored trial table is missing columns: {missing}")
    if not score_columns or len(score_columns) != len(set(score_columns)):
        raise ValueError("score_columns must be non-empty and unique.")
    grouping = [
        column for column in ("participant", "condition", "model")
        if column in scored_trials
    ]
    aggregation_rules = {
        "choice_log_score": "mean",
        "brier_score": "mean",
        "choice_correct": "mean",
        "rt_log_score": "mean",
        "absolute_log_rt_error": "mean",
        "absolute_rt_error_seconds": "median",
    }
    unknown = sorted(set(score_columns) - set(aggregation_rules))
    if unknown:
        raise ValueError(f"No frozen aggregation rule for score columns: {unknown}")
    for column in grouping:
        values = scored_trials[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Grouping column {column!r} contains missing/empty labels.")
    numeric = scored_trials.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() & ~scored_trials.loc[:, score_columns].isna()
    if invalid.any().any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Score columns may contain finite values or NaN only.")
    prepared = scored_trials.copy()
    prepared.loc[:, score_columns] = numeric
    return prepared.groupby(grouping, as_index=False, sort=True).agg(
        {column: aggregation_rules[column] for column in score_columns}
    )


def support_shift_flags(run_a_predictor: Any, run_b_predictor: Any) -> np.ndarray:
    """Flag B predictors outside the inclusive range observed in A."""

    run_a = _one_dimensional("run_a_predictor", run_a_predictor, dtype=float)
    run_b = _one_dimensional("run_b_predictor", run_b_predictor, dtype=float)
    if np.any(~np.isfinite(run_a)) or np.any(~np.isfinite(run_b)):
        raise ValueError("Support-shift predictors must be finite.")
    if run_a.size == 0 or run_b.size == 0:
        raise ValueError("Run-A and run-B predictor vectors cannot be empty.")
    return (run_b < np.min(run_a)) | (run_b > np.max(run_a))


def paired_participant_bootstrap(
    participant: Any,
    values_a: Any,
    values_b: Any,
    *,
    n_boot: int,
    seed: int,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap the paired A-B difference by resampling participant IDs."""

    ids = _one_dimensional("participant", participant, dtype=object)
    a = _one_dimensional("values_a", values_a, dtype=float)
    b = _one_dimensional("values_b", values_b, dtype=float)
    if not (ids.size == a.size == b.size):
        raise ValueError("Participant and paired-value vectors must have matching lengths.")
    if pd.isna(ids).any() or np.any([not str(value).strip() for value in ids]):
        raise ValueError("Participant IDs must be non-missing and non-empty.")
    n_resamples = _positive_integer("n_boot", n_boot)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one.")
    frame = pd.DataFrame(
        {
            "participant": np.asarray([str(value) for value in ids]),
            "a": a,
            "b": b,
        }
    ).dropna()
    if np.isinf(frame[["a", "b"]].to_numpy(dtype=float)).any():
        raise ValueError("Paired values must be finite or NaN.")
    # Duplicate IDs are allowed so all rows belonging to one participant can
    # travel together. For this scalar contrast, collapse within participant
    # before resampling IDs; subset by condition/model before calling when a
    # condition- or model-specific contrast is required.
    participant_pairs = frame.groupby("participant", as_index=False)[["a", "b"]].mean()
    differences = (participant_pairs["a"] - participant_pairs["b"]).to_numpy(dtype=float)
    if differences.size < 2:
        raise ValueError("Need at least two participants and one bootstrap resample.")
    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.choice(differences, size=differences.size, replace=True)
        resampled[index] = _finite_statistic(statistic, sample, context="a resample")
    alpha = 1.0 - float(confidence_level)
    return {
        "estimate": _finite_statistic(statistic, differences, context="the observed data"),
        "ci_low": float(np.quantile(resampled, alpha / 2.0)),
        "ci_high": float(np.quantile(resampled, 1.0 - alpha / 2.0)),
        "n_participants": int(differences.size),
    }


def participant_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    participant_column: str,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Resample participant IDs while carrying every condition/model row together."""

    if participant_column not in frame:
        raise ValueError(f"Missing participant column {participant_column!r}.")
    raw_ids = frame[participant_column]
    if raw_ids.isna().any() or raw_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Participant IDs must be non-missing and non-empty.")
    n_resamples = _positive_integer("n_boot", n_boot)
    participants = raw_ids.astype(str).unique()
    if participants.size < 2:
        raise ValueError("Need at least two participants and one bootstrap resample.")
    point = _finite_statistic(statistic, frame.copy(), context="the observed data")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    normalized = frame.copy()
    normalized[participant_column] = normalized[participant_column].astype(str)
    for index in range(n_resamples):
        sampled_ids = rng.choice(participants, size=participants.size, replace=True)
        pieces = []
        for draw, participant in enumerate(sampled_ids):
            participant_rows = normalized.loc[
                normalized[participant_column] == participant
            ].copy()
            # Give repeated draws distinct cluster labels while retaining the
            # source ID for any statistic that needs it.
            participant_rows["_source_participant"] = participant
            participant_rows[participant_column] = f"bootstrap_{draw}"
            pieces.append(participant_rows)
        boot[index] = _finite_statistic(
            statistic, pd.concat(pieces, ignore_index=True), context="a resample"
        )
    return point, boot


def holm_adjust(p_values: Any) -> np.ndarray:
    """Holm family-wise error adjustment, returned in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1].")
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * sorted_values
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
