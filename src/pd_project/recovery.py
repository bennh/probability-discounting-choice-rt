"""Synthetic-data generation and parameter-recovery summaries."""

from __future__ import annotations

from numbers import Integral
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


RECOVERY_PARAMETER_NAMES = {
    "log_k_R",
    "log_k_L",
    "log_beta",
    "alpha",
    "delta_L",
    "log_b",
    "log_sigma",
}


def _finite_scalar(name: str, value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _validated_range(name: str, bound: Any) -> tuple[float, float]:
    if not isinstance(bound, (list, tuple)) or len(bound) != 2:
        raise ValueError(f"Generating range {name!r} must be [lower, upper].")
    lower = _finite_scalar(f"{name} lower bound", bound[0])
    upper = _finite_scalar(f"{name} upper bound", bound[1])
    if not lower < upper:
        raise ValueError(f"Invalid generating range for {name}.")
    return lower, upper


def _strict_boolean_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype.")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array.astype(bool, copy=False)


def simulate_participant(
    design: pd.DataFrame,
    true_parameters: dict[str, float],
    model_name: str,
    *,
    rng: np.random.Generator,
    s0: float,
) -> pd.DataFrame:
    """Simulate choices and RTs on a fixed real or synthetic trial design."""

    required_columns = {"participant", "run", "condition", "r_cert", "r_uncert", "odds"}
    missing_columns = sorted(required_columns - set(design.columns))
    if missing_columns:
        raise ValueError(f"Simulation design is missing columns: {missing_columns}")
    if design.empty:
        raise ValueError("Simulation design must contain at least one trial.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")
    missing = sorted(RECOVERY_PARAMETER_NAMES - set(true_parameters))
    if missing:
        raise ValueError(f"Missing true simulation parameters: {missing}")
    parameters = {
        name: _finite_scalar(f"true parameter {name}", true_parameters[name])
        for name in RECOVERY_PARAMETER_NAMES
    }
    s0_scalar = _finite_scalar("s0", s0)
    if s0_scalar <= 0.0:
        raise ValueError("s0 must be positive.")
    simulated = design.copy(deep=True)
    conditions = simulated["condition"].to_numpy(dtype=str)
    with np.errstate(over="raise", invalid="raise"):
        try:
            k_reward = float(np.exp(parameters["log_k_R"]))
            k_loss = float(np.exp(parameters["log_k_L"]))
            beta = float(np.exp(parameters["log_beta"]))
            b = float(np.exp(parameters["log_b"]))
            sigma = float(np.exp(parameters["log_sigma"]))
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Exponentiated simulation parameters exceeded the finite numeric range."
            ) from exc
    k = parameter_by_condition(conditions, k_reward, k_loss)
    values = subjective_values(
        simulated["r_cert"].to_numpy(dtype=float),
        simulated["r_uncert"].to_numpy(dtype=float),
        simulated["odds"].to_numpy(dtype=float),
        k,
        s0=s0_scalar,
    )
    logits = choice_logits(beta, values.delta_v)
    predictor = rt_predictor(
        model_name, values.v_cert, values.v_uncert, values.delta_v
    )
    location = rt_location(
        parameters["alpha"],
        parameters["delta_L"],
        b,
        conditions,
        predictor,
    )
    probability = choice_probability(logits)
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise FloatingPointError("Simulation produced invalid choice probabilities.")
    n_trials = len(simulated)
    simulated["choice_uncertain"] = rng.binomial(1, probability, size=n_trials).astype(float)
    simulated["raw_action"] = simulated["choice_uncertain"] + 1.0
    with np.errstate(over="raise", invalid="raise"):
        try:
            generated_rt = rng.lognormal(mean=location, sigma=sigma, size=n_trials)
        except FloatingPointError as exc:
            raise FloatingPointError(
                "RT simulation exceeded the finite numeric range."
            ) from exc
    if np.any(~np.isfinite(generated_rt)) or np.any(generated_rt <= 0.0):
        raise FloatingPointError("Simulation produced non-finite or non-positive RT values.")
    simulated["rt_seconds"] = generated_rt
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

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")
    if not generating_ranges:
        raise ValueError("generating_ranges cannot be empty.")
    validated = {
        name: _validated_range(name, bound)
        for name, bound in generating_ranges.items()
        if name != "log_b"
    }
    if len(validated) != len(generating_ranges):
        raise ValueError("Pass log_b through log_b_range, not generating_ranges.")
    parameters = {
        name: float(rng.uniform(lower, upper))
        for name, (lower, upper) in validated.items()
    }
    lower_b, upper_b = _validated_range("log_b", log_b_range)
    parameters["log_b"] = float(rng.uniform(lower_b, upper_b))
    return parameters


def latin_hypercube_parameters(
    generating_ranges: dict[str, list[float]],
    *,
    n_samples: int,
    rng: np.random.Generator,
    log_b_range: tuple[float, float],
) -> list[dict[str, float]]:
    """Generate a stratified Latin-hypercube sample over all model parameters."""

    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")
    if not generating_ranges:
        raise ValueError("generating_ranges cannot be empty.")
    if "log_b" in generating_ranges:
        raise ValueError("Pass log_b through log_b_range, not generating_ranges.")
    ranges = {
        name: _validated_range(name, bound)
        for name, bound in generating_ranges.items()
    }
    ranges["log_b"] = _validated_range("log_b", log_b_range)
    draws: dict[str, np.ndarray] = {}
    for name, (lower, upper) in ranges.items():
        strata = (np.arange(n_samples, dtype=float) + rng.random(n_samples)) / n_samples
        draws[name] = lower + (upper - lower) * strata[rng.permutation(n_samples)]
    return [
        {name: float(values[index]) for name, values in draws.items()}
        for index in range(n_samples)
    ]


def recovery_metrics(
    true_values: Any,
    estimated_values: Any,
    *,
    parameter_range: float | None = None,
    success_mask: Any | None = None,
) -> dict[str, float | int | bool]:
    """Compute core recovery metrics for one parameter."""

    truth = np.asarray(true_values, dtype=float)
    estimate = np.asarray(estimated_values, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1:
        raise ValueError("true_values and estimated_values must be matching 1-D arrays.")
    if np.any(~np.isfinite(truth)):
        raise ValueError("All declared true parameter values must be finite.")
    if success_mask is None:
        success = np.ones(truth.shape, dtype=bool)
        if np.any(~np.isfinite(estimate)):
            raise ValueError(
                "Non-finite estimates require an explicit success_mask; "
                "failed fits cannot be silently discarded."
            )
    else:
        success = _strict_boolean_array("success_mask", success_mask, truth.shape)
    if np.any(~np.isfinite(estimate[success])):
        raise ValueError("Every successful fit must have a finite estimate.")
    if int(success.sum()) < 2:
        raise ValueError("At least two successful recovery fits are required.")
    successful_truth = truth[success]
    successful_estimate = estimate[success]
    if np.ptp(successful_truth) <= 0.0:
        raise ValueError("Successful true parameter values must have positive variation.")
    errors = successful_estimate - successful_truth
    slope, intercept = np.polyfit(successful_truth, successful_estimate, deg=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    if parameter_range is None:
        raise ValueError(
            "parameter_range must be the positive width of the pre-declared generating range."
        )
    denominator = _finite_scalar("parameter_range", parameter_range)
    if denominator <= 0.0:
        raise ValueError("parameter_range must be finite and positive.")
    estimate_varies = bool(np.ptp(successful_estimate) > 0.0)
    correlation = (
        float(np.corrcoef(successful_truth, successful_estimate)[0, 1])
        if estimate_varies
        else float("nan")
    )
    return {
        "n_total": int(truth.size),
        "n_success": int(success.sum()),
        "optimizer_failure_rate": float(1.0 - success.mean()),
        "bias": float(np.mean(errors)),
        "rmse": rmse,
        "nrmse": float(rmse / denominator),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "correlation": correlation,
        "correlation_defined": estimate_varies,
    }


def model_recovery_confusion(
    results: pd.DataFrame,
    *,
    generating_column: str = "generating_model",
    fitted_column: str = "fitted_model",
    score_column: str = "run_b_rt_mlpd",
    expected_replicates: int | None = None,
) -> pd.DataFrame:
    """Select the best held-out RT model per replicate and form a row-normalized matrix."""

    required = {"replicate", generating_column, fitted_column, score_column}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Model-recovery table is missing columns: {missing}")
    working = results.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    if not np.all(np.isfinite(working[score_column].to_numpy(dtype=float))):
        raise ValueError("Model-recovery scores must all be finite.")
    working[generating_column] = working[generating_column].astype(str)
    working[fitted_column] = working[fitted_column].astype(str)
    expected_models = {"M1", "M2", "M3"}
    if set(working[generating_column]) != expected_models:
        raise ValueError("Model recovery requires generating models M1/M2/M3.")
    if set(working[fitted_column]) != expected_models:
        raise ValueError("Model recovery requires fitted models M1/M2/M3.")
    replicate_sets = {
        model: set(group["replicate"])
        for model, group in working.groupby(generating_column)
    }
    if len({frozenset(values) for values in replicate_sets.values()}) != 1:
        raise ValueError("Generating models must use exactly the same replicate IDs.")
    replicate_ids = next(iter(replicate_sets.values()))
    if expected_replicates is not None:
        if (
            isinstance(expected_replicates, bool)
            or not isinstance(expected_replicates, Integral)
            or expected_replicates < 1
        ):
            raise ValueError("expected_replicates must be a positive integer.")
        if len(replicate_ids) != expected_replicates:
            raise ValueError(
                f"Expected {expected_replicates} replicates per generating model; "
                f"found {len(replicate_ids)}."
            )
    unit_columns = [
        name
        for name in ("participant", "run", "condition", "trial_index")
        if name in working.columns
    ]
    for (generating_model, replicate), group in working.groupby(
        [generating_column, "replicate"]
    ):
        fitted_groups = {
            model: model_group
            for model, model_group in group.groupby(fitted_column)
        }
        if set(fitted_groups) != expected_models:
            raise ValueError(
                f"Missing fitted model for {generating_model}/{replicate}."
            )
        counts = {model: len(model_group) for model, model_group in fitted_groups.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                "Fitted models must score the same number of held-out units; "
                f"got {counts} for {generating_model}/{replicate}."
            )
        if unit_columns:
            key_sets = {}
            for model, model_group in fitted_groups.items():
                keys = list(model_group[unit_columns].itertuples(index=False, name=None))
                if len(keys) != len(set(keys)):
                    raise ValueError(
                        f"Duplicate held-out score keys for {generating_model}/"
                        f"{replicate}/{model}."
                    )
                key_sets[model] = set(keys)
            if len({frozenset(keys) for keys in key_sets.values()}) != 1:
                raise ValueError(
                    "Fitted models must score exactly the same held-out units for "
                    f"{generating_model}/{replicate}."
                )
    aggregated = (
        working.groupby(
            [generating_column, "replicate", fitted_column], as_index=False
        )[score_column]
        .mean()
    )
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
