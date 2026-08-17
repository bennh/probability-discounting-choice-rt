"""Test-retest reliability and agreement diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def icc_a1(values: Any) -> float:
    """ICC(A,1): two-way random, absolute agreement, single measurement."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("ICC requires participants x runs with at least 2 of each.")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("ICC input must be complete and finite.")
    n_participants, n_runs = matrix.shape
    grand_mean = matrix.mean()
    participant_means = matrix.mean(axis=1)
    run_means = matrix.mean(axis=0)
    ms_participant = n_runs * np.sum((participant_means - grand_mean) ** 2) / (
        n_participants - 1
    )
    ms_run = n_participants * np.sum((run_means - grand_mean) ** 2) / (n_runs - 1)
    residual = (
        matrix
        - participant_means[:, None]
        - run_means[None, :]
        + grand_mean
    )
    ms_error = np.sum(residual**2) / ((n_participants - 1) * (n_runs - 1))
    denominator = (
        ms_participant
        + (n_runs - 1) * ms_error
        + (n_runs / n_participants) * (ms_run - ms_error)
    )
    # Do not use an absolute-tolerance comparison here: ICC must be invariant
    # to multiplying every measurement by a small non-zero constant.
    if denominator == 0.0:
        return np.nan
    result = float((ms_participant - ms_error) / denominator)
    if not np.isfinite(result):
        raise FloatingPointError("ICC computation produced a non-finite value.")
    return result


def bland_altman_summary(run_a: Any, run_b: Any) -> dict[str, float]:
    a = np.asarray(run_a, dtype=float)
    b = np.asarray(run_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Bland-Altman inputs must be matching one-dimensional arrays.")
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        raise ValueError("At least two finite A/B pairs are required.")
    difference = b[valid] - a[valid]
    mean_shift = float(np.mean(difference))
    standard_deviation = float(np.std(difference, ddof=1))
    return {
        "n": int(valid.sum()),
        "mean_shift_b_minus_a": mean_shift,
        "lower_limit_of_agreement": mean_shift - 1.96 * standard_deviation,
        "upper_limit_of_agreement": mean_shift + 1.96 * standard_deviation,
    }


def paired_bootstrap_icc_difference(
    estimates: pd.DataFrame,
    *,
    participant_column: str,
    full_a_column: str,
    full_b_column: str,
    choice_a_column: str,
    choice_b_column: str,
    n_boot: int,
    seed: int,
    minimum_valid_fraction: float = 0.90,
) -> dict[str, float]:
    required = {
        participant_column,
        full_a_column,
        full_b_column,
        choice_a_column,
        choice_b_column,
    }
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError(f"ICC bootstrap table is missing columns: {missing}")
    n_resamples = _positive_integer("n_boot", n_boot)
    if (
        not np.isfinite(minimum_valid_fraction)
        or not 0.0 < minimum_valid_fraction <= 1.0
    ):
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")
    identifiers = estimates[participant_column]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("Participant IDs must be non-missing and non-empty.")
    normalized_ids = identifiers.astype(str)
    if normalized_ids.duplicated().any():
        raise ValueError("ICC bootstrap requires one keyed row per participant.")
    prepared = estimates.copy()
    prepared[participant_column] = normalized_ids
    # Canonical ordering makes a seeded bootstrap invariant to input row order.
    prepared = prepared.sort_values(participant_column, kind="stable").reset_index(drop=True)
    arrays = [
        pd.to_numeric(prepared[column], errors="raise").to_numpy(dtype=float)
        for column in (full_a_column, full_b_column, choice_a_column, choice_b_column)
    ]
    if any(np.isinf(array).any() for array in arrays):
        raise ValueError("ICC estimates must be finite or NaN; infinity is invalid.")
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    arrays = [array[valid] for array in arrays]
    if arrays[0].size < 3:
        raise ValueError("At least three complete participants are required.")
    observed = icc_a1(np.column_stack(arrays[:2])) - icc_a1(
        np.column_stack(arrays[2:])
    )
    if not np.isfinite(observed):
        raise RuntimeError("Observed ICC difference is undefined.")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, arrays[0].size, size=arrays[0].size)
        boot[index] = icc_a1(np.column_stack((arrays[0][sample], arrays[1][sample]))) - icc_a1(
            np.column_stack((arrays[2][sample], arrays[3][sample]))
        )
    finite = boot[np.isfinite(boot)]
    required_valid = int(np.ceil(n_resamples * minimum_valid_fraction))
    if finite.size < required_valid:
        raise RuntimeError(
            f"Only {finite.size}/{n_resamples} bootstrap ICC differences were defined; "
            f"required at least {required_valid}."
        )
    return {
        "estimate": float(observed),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "valid_bootstrap_resamples": int(finite.size),
        "n_complete_participants": int(arrays[0].size),
    }
