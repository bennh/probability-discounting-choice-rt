"""Participant-level bounded multistart maximum-likelihood fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .likelihood import bernoulli_logpmf_from_logits, joint_log_likelihood
from .rt_models import rt_location, rt_predictor
from .valuation_choice import choice_logits, parameter_by_condition, subjective_values


CHOICE_PARAMETER_NAMES = ("log_k_R", "log_k_L", "log_beta")
FULL_PARAMETER_NAMES = CHOICE_PARAMETER_NAMES + (
    "alpha",
    "delta_L",
    "log_b",
    "log_sigma",
)
INVALID_OBJECTIVE = 1.0e100


def _strict_boolean_column(trials: pd.DataFrame, name: str) -> np.ndarray:
    if name not in trials.columns:
        raise ValueError(f"Trials are missing required boolean column {name!r}.")
    values = trials[name].to_numpy(copy=False)
    if values.dtype != np.dtype(bool):
        raise ValueError(f"Trial column {name!r} must have boolean dtype.")
    return values.astype(bool, copy=False)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_option(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(array)
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}.")
    if minimum is not None and scalar < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return scalar


@dataclass(frozen=True)
class StartDiagnostic:
    start_index: int
    objective: float
    optimizer_success: bool
    valid: bool
    message: str
    iterations: int | None
    gradient_norm: float | None
    projected_gradient_norm: float | None
    near_boundary: bool
    estimates: tuple[float, ...]


@dataclass(frozen=True)
class FitResult:
    participant: str
    model: str
    choice_only: bool
    parameter_names: tuple[str, ...]
    estimates: dict[str, float]
    objective: float
    success: bool
    best_start_index: int
    message: str
    starts: tuple[StartDiagnostic, ...] = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "participant": self.participant,
            "model": self.model,
            "choice_only": self.choice_only,
            "objective": self.objective,
            "success": self.success,
            "best_start_index": self.best_start_index,
            "message": self.message,
        }
        record.update(self.estimates)
        return record


def _bounds_for(
    config: dict[str, Any], model_name: str, choice_only: bool
) -> tuple[tuple[str, ...], list[tuple[float, float]]]:
    names = CHOICE_PARAMETER_NAMES if choice_only else FULL_PARAMETER_NAMES
    configured = config["optimization"]["bounds"]
    bounds: list[tuple[float, float]] = []
    for name in names:
        if name == "log_b":
            pair = configured["log_b"][model_name.upper()]
        else:
            pair = configured[name]
        bounds.append((float(pair[0]), float(pair[1])))
    return names, bounds


def _objective(
    vector: np.ndarray,
    trials: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    choice_only: bool,
) -> float:
    try:
        parameters = dict(
            zip(CHOICE_PARAMETER_NAMES if choice_only else FULL_PARAMETER_NAMES, vector)
        )
        conditions = trials["condition"].to_numpy(dtype=str)
        k = parameter_by_condition(
            conditions,
            np.exp(parameters["log_k_R"]),
            np.exp(parameters["log_k_L"]),
        )
        scale = float(config["data"]["transforms"]["amount_scale"])
        values = subjective_values(
            trials["r_cert"].to_numpy(dtype=float),
            trials["r_uncert"].to_numpy(dtype=float),
            trials["odds"].to_numpy(dtype=float),
            k,
            s0=scale,
        )
        logits = choice_logits(np.exp(parameters["log_beta"]), values.delta_v)
        include_choice = _strict_boolean_column(trials, "choice_included")
        choices = trials["choice_uncertain"].to_numpy(dtype=float)
        if choice_only:
            log_likelihood = bernoulli_logpmf_from_logits(
                choices[include_choice], logits[include_choice]
            ).sum()
            return -float(log_likelihood)

        predictor = rt_predictor(
            model_name, values.v_cert, values.v_uncert, values.delta_v
        )
        location = rt_location(
            parameters["alpha"],
            parameters["delta_L"],
            np.exp(parameters["log_b"]),
            conditions,
            predictor,
        )
        log_likelihood = joint_log_likelihood(
            choices,
            trials["rt_seconds"].to_numpy(dtype=float),
            logits,
            location,
            np.exp(parameters["log_sigma"]),
            choice_mask=include_choice,
            rt_mask=_strict_boolean_column(trials, "rt_included"),
        )
        return -log_likelihood
    except (ValueError, FloatingPointError, OverflowError):
        return INVALID_OBJECTIVE


def _near_boundary(
    estimates: np.ndarray,
    bounds: list[tuple[float, float]],
    fraction: float,
) -> bool:
    for estimate, (lower, upper) in zip(estimates, bounds):
        margin = fraction * (upper - lower)
        if estimate <= lower + margin or estimate >= upper - margin:
            return True
    return False


def _projected_gradient_inf_norm(
    estimates: np.ndarray,
    gradient: np.ndarray | None,
    bounds: list[tuple[float, float]],
) -> float | None:
    if gradient is None or not np.all(np.isfinite(gradient)):
        return None
    projected = np.asarray(gradient, dtype=float).copy()
    for index, (estimate, (lower, upper)) in enumerate(zip(estimates, bounds)):
        tolerance = 1.0e-10 * max(1.0, upper - lower)
        if estimate <= lower + tolerance and projected[index] > 0.0:
            projected[index] = 0.0
        elif estimate >= upper - tolerance and projected[index] < 0.0:
            projected[index] = 0.0
    return float(np.linalg.norm(projected, ord=np.inf))


def fit_participant(
    trials: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    *,
    seed: int,
    choice_only: bool = False,
    multistarts: int | None = None,
) -> FitResult:
    """Fit one participant using deterministic bounded L-BFGS-B starts."""

    model = model_name.upper()
    if choice_only and model != "M1":
        raise ValueError(
            "The single choice-only baseline must use canonical model_name='M1'."
        )
    if not choice_only and model not in ("M1", "M2", "M3"):
        raise ValueError("Full fits require model_name M1, M2, or M3.")
    participants = trials["participant"].astype(str).unique()
    if len(participants) != 1:
        raise ValueError("fit_participant expects trials from exactly one participant.")
    choice_included = _strict_boolean_column(trials, "choice_included")
    rt_included = _strict_boolean_column(trials, "rt_included")
    if np.any(rt_included & ~choice_included):
        raise ValueError("Every RT-included trial must also have a valid choice.")
    if not bool(choice_included.any()):
        raise ValueError("Participant has no valid choices.")
    if not choice_only and not bool(rt_included.any()):
        raise ValueError("Participant has no valid RT observations.")

    names, bounds = _bounds_for(config, model, choice_only)
    n_starts = _positive_integer(
        "multistarts",
        config["optimization"]["multistarts"] if multistarts is None else multistarts
    )
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng = np.random.default_rng(seed)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [0.5 * (lower + upper)]
    starts.extend(rng.uniform(lower, upper) for _ in range(n_starts - 1))

    optimization = config["optimization"]
    method = optimization.get("method")
    if method != "L-BFGS-B":
        raise ValueError("The frozen optimization method must be 'L-BFGS-B'.")
    valid_fit = optimization.get("valid_fit", {})
    require_success = valid_fit.get("require_optimizer_success", True)
    if not isinstance(require_success, bool):
        raise ValueError("require_optimizer_success must be a boolean.")
    if valid_fit.get("require_finite_objective", True) is not True:
        raise ValueError("require_finite_objective must remain true.")
    boundary_fraction = _finite_option(
        "boundary_near_fraction",
        valid_fit.get("boundary_near_fraction", 0.01),
        minimum=0.0,
        maximum=0.5,
    )
    max_iterations = _positive_integer(
        "max_iterations", optimization.get("max_iterations", 5000)
    )
    ftol = _finite_option("ftol", optimization.get("ftol", 1.0e-10), positive=True)
    gtol = _finite_option("gtol", optimization.get("gtol", 1.0e-6), positive=True)
    projected_threshold = valid_fit.get("projected_gradient_inf_max")
    if projected_threshold is not None:
        projected_threshold = _finite_option(
            "projected_gradient_inf_max", projected_threshold, minimum=0.0
        )
    diagnostics: list[StartDiagnostic] = []
    for index, start in enumerate(starts):
        result = minimize(
            _objective,
            np.asarray(start, dtype=float),
            args=(trials, model, config, choice_only),
            method=method,
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "ftol": ftol,
                "gtol": gtol,
            },
        )
        finite = bool(
            np.isfinite(result.fun)
            and float(result.fun) < INVALID_OBJECTIVE / 10.0
            and np.all(np.isfinite(result.x))
        )
        jacobian = getattr(result, "jac", None)
        gradient_norm = (
            float(np.linalg.norm(jacobian, ord=np.inf))
            if jacobian is not None and np.all(np.isfinite(jacobian))
            else None
        )
        projected_gradient_norm = _projected_gradient_inf_norm(
            result.x, jacobian, bounds
        )
        gradient_valid = (
            projected_gradient_norm is not None
            and (
                projected_threshold is None
                or projected_gradient_norm <= float(projected_threshold)
            )
        )
        valid = (
            finite
            and (bool(result.success) or not require_success)
            and gradient_valid
        )
        diagnostic = StartDiagnostic(
            start_index=index,
            objective=float(result.fun),
            optimizer_success=bool(result.success),
            valid=valid,
            message=str(result.message),
            iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
            gradient_norm=gradient_norm,
            projected_gradient_norm=projected_gradient_norm,
            near_boundary=_near_boundary(result.x, bounds, boundary_fraction),
            estimates=tuple(float(value) for value in result.x),
        )
        diagnostics.append(diagnostic)

    valid_indices = [index for index, item in enumerate(diagnostics) if item.valid]
    candidate_indices = valid_indices or [
        index
        for index, item in enumerate(diagnostics)
        if np.isfinite(item.objective)
        and item.objective < INVALID_OBJECTIVE / 10.0
        and np.all(np.isfinite(item.estimates))
    ]
    if not candidate_indices:
        return FitResult(
            participant=str(participants[0]),
            model="choice_only" if choice_only else model,
            choice_only=choice_only,
            parameter_names=names,
            estimates={name: float("nan") for name in names},
            objective=INVALID_OBJECTIVE,
            success=False,
            best_start_index=-1,
            message="Every optimization start returned an invalid or penalized solution.",
            starts=tuple(diagnostics),
        )
    best_index = min(candidate_indices, key=lambda index: diagnostics[index].objective)
    best = diagnostics[best_index]
    estimates = dict(zip(names, best.estimates))
    return FitResult(
        participant=str(participants[0]),
        model="choice_only" if choice_only else model,
        choice_only=choice_only,
        parameter_names=names,
        estimates=estimates,
        objective=best.objective,
        success=best.valid,
        best_start_index=best_index,
        message=best.message,
        starts=tuple(diagnostics),
    )


def deterministic_seed(master_seed: int, *keys: Any) -> int:
    """Derive a stable 32-bit seed independent of Python's randomized hash."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, Integral):
        raise ValueError("master_seed must be an integer.")
    payload = json.dumps(
        [int(master_seed), *(str(key) for key in keys)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def fit_dataset(
    trials: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    *,
    choice_only: bool = False,
    multistarts: int | None = None,
) -> list[FitResult]:
    """Fit all participants independently in sorted participant order."""

    master_seed = int(config["random"]["master_seed"])
    results: list[FitResult] = []
    if "run" in trials:
        unique_runs = sorted(trials["run"].astype(str).unique())
        if len(unique_runs) != 1:
            raise ValueError(
                "fit_dataset requires exactly one run at a time; fit A and B independently."
            )
        run_key = unique_runs[0]
    else:
        run_key = "unspecified"
    for participant, participant_trials in trials.groupby("participant", sort=True):
        seed_model_key = "choice_only" if choice_only else model_name.upper()
        seed = deterministic_seed(
            master_seed,
            "fit",
            participant,
            run_key,
            seed_model_key,
            "choice" if choice_only else "full",
        )
        results.append(
            fit_participant(
                participant_trials,
                model_name,
                config,
                seed=seed,
                choice_only=choice_only,
                multistarts=multistarts,
            )
        )
    return results


def fit_results_frame(results: list[FitResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_record() for result in results])


def validate_run_a_fit_artifact(
    fits: pd.DataFrame,
    participants: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and partition the exact run-A fit matrix before held-out access."""

    base_columns = {"participant", "model", "choice_only", "success", "objective"}
    missing_base = sorted(base_columns - set(fits.columns))
    if missing_base:
        raise ValueError(f"Run-A fit artifact is missing columns: {missing_base}")

    def strict_boolean(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"Invalid boolean value in Run-A fit artifact: {value!r}")

    choice_only = fits["choice_only"].map(strict_boolean)
    success = fits["success"].map(strict_boolean)
    models = fits["model"].astype(str)
    valid_models = {"choice_only", "M1", "M2", "M3"}
    if not set(models).issubset(valid_models):
        raise ValueError("Run-A fit artifact contains an unknown or non-canonical model label.")
    expected_choice_only = models.eq("choice_only")
    if not bool(choice_only.eq(expected_choice_only).all()):
        raise ValueError("Run-A model labels disagree with the choice_only flag.")
    if not bool(success.all()):
        failed = fits.loc[~success, ["participant", "model"]]
        raise ValueError(
            "Run-A fit artifact contains unsuccessful fits: "
            f"{failed.to_dict(orient='records')}"
        )

    objectives = pd.to_numeric(fits["objective"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(objectives)):
        raise ValueError("Run-A fit objectives must all be finite numeric values.")

    required_parameter_columns = set(FULL_PARAMETER_NAMES)
    missing_parameters = sorted(required_parameter_columns - set(fits.columns))
    if missing_parameters:
        raise ValueError(f"Run-A fit artifact is missing parameters: {missing_parameters}")
    for index, row in fits.iterrows():
        model = str(row["model"])
        names = CHOICE_PARAMETER_NAMES if model == "choice_only" else FULL_PARAMETER_NAMES
        values = pd.to_numeric(row[list(names)], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Run-A fit parameters must be finite for row {index}, model {model}."
            )
        bound_model = "M1" if model == "choice_only" else model
        _, bounds = _bounds_for(config, bound_model, model == "choice_only")
        for name, value, (lower, upper) in zip(names, values, bounds):
            if value < lower or value > upper:
                raise ValueError(
                    f"Run-A parameter {name}={value} is outside [{lower}, {upper}] "
                    f"for row {index}, model {model}."
                )

    participant_values = fits["participant"].astype(str)
    expected_pairs = {
        (participant, model)
        for participant in participants
        for model in ("choice_only", "M1", "M2", "M3")
    }
    pair_counts = (
        fits.assign(participant=participant_values, model=models)
        .groupby(["participant", "model"])
        .size()
    )
    observed_pairs = set(pair_counts.index)
    nonunique = pair_counts.loc[pair_counts != 1]
    if observed_pairs != expected_pairs or not nonunique.empty:
        raise ValueError(
            "Run-A fit matrix must contain exactly one choice_only/M1/M2/M3 row "
            f"per participant. Missing={sorted(expected_pairs - observed_pairs)}, "
            f"extra={sorted(observed_pairs - expected_pairs)}, "
            f"nonunique={nonunique.to_dict()}"
        )

    normalized = fits.copy()
    normalized["choice_only"] = choice_only.to_numpy(dtype=bool)
    normalized["success"] = success.to_numpy(dtype=bool)
    return (
        normalized.loc[~normalized["choice_only"]].copy(),
        normalized.loc[normalized["choice_only"]].copy(),
    )


def predict_from_estimates(
    trials: pd.DataFrame,
    estimates: dict[str, float],
    model_name: str,
    config: dict[str, Any],
    *,
    choice_only: bool = False,
) -> pd.DataFrame:
    """Create trial-level latent values and predictive parameters from a fit."""

    required = set(CHOICE_PARAMETER_NAMES if choice_only else FULL_PARAMETER_NAMES)
    missing = sorted(required - set(estimates))
    if missing:
        raise ValueError(f"Fit is missing parameters required for prediction: {missing}")
    conditions = trials["condition"].to_numpy(dtype=str)
    k = parameter_by_condition(
        conditions,
        np.exp(estimates["log_k_R"]),
        np.exp(estimates["log_k_L"]),
    )
    scale = float(config["data"]["transforms"]["amount_scale"])
    values = subjective_values(
        trials["r_cert"].to_numpy(dtype=float),
        trials["r_uncert"].to_numpy(dtype=float),
        trials["odds"].to_numpy(dtype=float),
        k,
        s0=scale,
    )
    logits = choice_logits(np.exp(estimates["log_beta"]), values.delta_v)
    output = pd.DataFrame(
        {
            "v_cert": values.v_cert,
            "v_uncert": values.v_uncert,
            "delta_v": values.delta_v,
            "choice_logit": logits,
        },
        index=trials.index,
    )
    if not choice_only:
        predictor = rt_predictor(
            model_name, values.v_cert, values.v_uncert, values.delta_v
        )
        output["rt_predictor"] = predictor
        output["rt_mu"] = rt_location(
            estimates["alpha"],
            estimates["delta_L"],
            np.exp(estimates["log_b"]),
            conditions,
            predictor,
        )
        output["rt_sigma"] = np.exp(estimates["log_sigma"])
    return output


def start_diagnostics_frame(results: list[FitResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit in results:
        for start in fit.starts:
            row = {
                "participant": fit.participant,
                "model": fit.model,
                "choice_only": fit.choice_only,
                **{
                    field_name: getattr(start, field_name)
                    for field_name in (
                        "start_index",
                        "objective",
                        "optimizer_success",
                        "valid",
                        "message",
                        "iterations",
                        "gradient_norm",
                        "projected_gradient_norm",
                        "near_boundary",
                    )
                },
            }
            row.update(dict(zip(fit.parameter_names, start.estimates)))
            rows.append(row)
    return pd.DataFrame(rows)
