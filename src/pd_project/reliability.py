"""Test-retest reliability and agreement diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
    return float((ms_participant - ms_error) / denominator)


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
    if estimates[participant_column].astype(str).duplicated().any():
        raise ValueError("ICC bootstrap requires one keyed row per participant.")
    arrays = [
        estimates[column].to_numpy(dtype=float)
        for column in (full_a_column, full_b_column, choice_a_column, choice_b_column)
    ]
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    arrays = [array[valid] for array in arrays]
    if arrays[0].size < 3:
        raise ValueError("At least three complete participants are required.")
    observed = icc_a1(np.column_stack(arrays[:2])) - icc_a1(
        np.column_stack(arrays[2:])
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, arrays[0].size, size=arrays[0].size)
        boot[index] = icc_a1(np.column_stack((arrays[0][sample], arrays[1][sample]))) - icc_a1(
            np.column_stack((arrays[2][sample], arrays[3][sample]))
        )
    finite = boot[np.isfinite(boot)]
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")
    required_valid = int(np.ceil(n_boot * minimum_valid_fraction))
    if finite.size < required_valid:
        raise RuntimeError(
            f"Only {finite.size}/{n_boot} bootstrap ICC differences were defined; "
            f"required at least {required_valid}."
        )
    return {
        "estimate": float(observed),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "valid_bootstrap_resamples": int(finite.size),
    }
