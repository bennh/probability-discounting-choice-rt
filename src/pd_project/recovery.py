"""Synthetic-data generation and parameter-recovery summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rt_models import rt_location, rt_predictor
from .valuation_choice import (
    choice_logits,
    choice_probability,
    parameter_by_condition,
    subjective_values,
)


def simulate_participant(
    design: pd.DataFrame,
    true_parameters: dict[str, float],
    model_name: str,
    *,
    rng: np.random.Generator,
    s0: float,
) -> pd.DataFrame:
    """Simulate choices and RTs on a fixed real or synthetic trial design."""

    required = {"log_k_R", "log_k_L", "log_beta", "alpha", "delta_L", "log_b", "log_sigma"}
    missing = sorted(required - set(true_parameters))
    if missing:
        raise ValueError(f"Missing true simulation parameters: {missing}")
    simulated = design.copy(deep=True)
    conditions = simulated["condition"].to_numpy(dtype=str)
    k = parameter_by_condition(
        conditions,
        np.exp(true_parameters["log_k_R"]),
        np.exp(true_parameters["log_k_L"]),
    )
    values = subjective_values(
        simulated["r_cert"].to_numpy(dtype=float),
        simulated["r_uncert"].to_numpy(dtype=float),
        simulated["odds"].to_numpy(dtype=float),
        k,
        s0=s0,
    )
    logits = choice_logits(np.exp(true_parameters["log_beta"]), values.delta_v)
    predictor = rt_predictor(
        model_name, values.v_cert, values.v_uncert, values.delta_v
    )
    location = rt_location(
        true_parameters["alpha"],
        true_parameters["delta_L"],
        np.exp(true_parameters["log_b"]),
        conditions,
        predictor,
    )
    probability = choice_probability(logits)
    n_trials = len(simulated)
    simulated["choice_uncertain"] = rng.binomial(1, probability, size=n_trials).astype(float)
    simulated["raw_action"] = simulated["choice_uncertain"] + 1.0
    simulated["rt_seconds"] = rng.lognormal(
        mean=location, sigma=np.exp(true_parameters["log_sigma"]), size=n_trials
    )
    simulated["choice_included"] = True
    simulated["rt_included"] = True
    return simulated


def sample_true_parameters(
    generating_ranges: dict[str, list[float]],
    *,
    rng: np.random.Generator,
    log_b_range: tuple[float, float] = (-2.0, 1.0),
) -> dict[str, float]:
    """Draw one iid diagnostic vector; formal recovery uses Latin hypercubes."""

    parameters = {
        name: float(rng.uniform(float(bound[0]), float(bound[1])))
        for name, bound in generating_ranges.items()
    }
    parameters["log_b"] = float(rng.uniform(*log_b_range))
    return parameters


def latin_hypercube_parameters(
    generating_ranges: dict[str, list[float]],
    *,
    n_samples: int,
    rng: np.random.Generator,
    log_b_range: tuple[float, float],
) -> list[dict[str, float]]:
    """Generate a stratified Latin-hypercube sample over all model parameters."""

    if n_samples < 1:
        raise ValueError("n_samples must be at least one.")
    ranges = {**generating_ranges, "log_b": list(log_b_range)}
    draws: dict[str, np.ndarray] = {}
    for name, bound in ranges.items():
        lower, upper = float(bound[0]), float(bound[1])
        if not lower < upper:
            raise ValueError(f"Invalid generating range for {name}.")
        strata = (np.arange(n_samples, dtype=float) + rng.random(n_samples)) / n_samples
        draws[name] = lower + (upper - lower) * strata[rng.permutation(n_samples)]
    return [
        {name: float(values[index]) for name, values in draws.items()}
        for index in range(n_samples)
    ]


def recovery_metrics(
    true_values: Any, estimated_values: Any, *, parameter_range: float | None = None
) -> dict[str, float]:
    """Compute core recovery metrics for one parameter."""

    truth = np.asarray(true_values, dtype=float)
    estimate = np.asarray(estimated_values, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(estimate)
    if valid.sum() < 2:
        raise ValueError("At least two finite true/estimated pairs are required.")
    truth = truth[valid]
    estimate = estimate[valid]
    errors = estimate - truth
    slope, intercept = np.polyfit(truth, estimate, deg=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    if parameter_range is None or not np.isfinite(parameter_range) or parameter_range <= 0.0:
        raise ValueError(
            "parameter_range must be the positive width of the pre-declared generating range."
        )
    denominator = float(parameter_range)
    return {
        "n": int(valid.sum()),
        "bias": float(np.mean(errors)),
        "rmse": rmse,
        "nrmse": float(rmse / denominator) if denominator > 0.0 else np.nan,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "correlation": float(np.corrcoef(truth, estimate)[0, 1]),
    }


def model_recovery_confusion(
    results: pd.DataFrame,
    *,
    generating_column: str = "generating_model",
    fitted_column: str = "fitted_model",
    score_column: str = "run_b_rt_mlpd",
) -> pd.DataFrame:
    """Select the best held-out RT model per replicate and form a row-normalized matrix."""

    required = {"replicate", generating_column, fitted_column, score_column}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Model-recovery table is missing columns: {missing}")
    if not np.all(np.isfinite(results[score_column].to_numpy(dtype=float))):
        raise ValueError("Model-recovery scores must all be finite.")
    aggregated = (
        results.groupby(
            [generating_column, "replicate", fitted_column], as_index=False
        )[score_column]
        .mean()
    )
    expected_models = {"M1", "M2", "M3"}
    observed_generating_models = set(aggregated[generating_column].astype(str))
    if observed_generating_models != expected_models:
        raise ValueError(
            "Model recovery requires generating models M1/M2/M3; got "
            f"{observed_generating_models}."
        )
    for (generating_model, replicate), group in aggregated.groupby(
        [generating_column, "replicate"]
    ):
        observed_models = set(group[fitted_column].astype(str))
        if observed_models != expected_models or len(group) != 3:
            raise ValueError(
                "Every generating-model replicate must contain exactly one aggregated "
                f"score for M1/M2/M3; got {observed_models} for "
                f"{generating_model}/{replicate}."
            )
        best_score = group[score_column].max()
        if int((group[score_column] == best_score).sum()) != 1:
            raise ValueError(
                f"Model-recovery winner is tied for {generating_model}/{replicate}; "
                "apply the pre-declared tie policy before building a confusion matrix."
            )
    winners = aggregated.loc[
        aggregated.groupby([generating_column, "replicate"])[score_column].idxmax()
    ]
    counts = pd.crosstab(
        winners[generating_column], winners[fitted_column], normalize="index"
    )
    return counts.reindex(index=["M1", "M2", "M3"], columns=["M1", "M2", "M3"], fill_value=0.0)
