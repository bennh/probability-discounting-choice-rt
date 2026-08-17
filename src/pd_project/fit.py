"""Participant-level bounded multistart maximum-likelihood fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
        include_choice = trials["choice_included"].to_numpy(dtype=bool)
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
            rt_mask=trials["rt_included"].to_numpy(dtype=bool),
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
    if not choice_only and model not in ("M1", "M2", "M3"):
        raise ValueError("Full fits require model_name M1, M2, or M3.")
    participants = trials["participant"].astype(str).unique()
    if len(participants) != 1:
        raise ValueError("fit_participant expects trials from exactly one participant.")
    if not bool(trials["choice_included"].any()):
        raise ValueError("Participant has no valid choices.")
    if not choice_only and not bool(trials["rt_included"].any()):
        raise ValueError("Participant has no valid RT observations.")

    names, bounds = _bounds_for(config, model, choice_only)
    n_starts = int(
        config["optimization"]["multistarts"] if multistarts is None else multistarts
    )
    if n_starts < 1:
        raise ValueError("multistarts must be at least one.")
    rng = np.random.default_rng(seed)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [0.5 * (lower + upper)]
    starts.extend(rng.uniform(lower, upper) for _ in range(n_starts - 1))

    optimization = config["optimization"]
    require_success = bool(
        optimization.get("valid_fit", {}).get("require_optimizer_success", True)
    )
    boundary_fraction = float(
        optimization.get("valid_fit", {}).get("boundary_near_fraction", 0.01)
    )
    diagnostics: list[StartDiagnostic] = []
    raw_results = []
    for index, start in enumerate(starts):
        result = minimize(
            _objective,
            np.asarray(start, dtype=float),
            args=(trials, model, config, choice_only),
            method=str(optimization.get("method", "L-BFGS-B")),
            bounds=bounds,
            options={
                "maxiter": int(optimization.get("max_iterations", 5000)),
                "ftol": float(optimization.get("ftol", 1.0e-10)),
                "gtol": float(optimization.get("gtol", 1.0e-6)),
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
        projected_threshold = optimization.get("valid_fit", {}).get(
            "projected_gradient_inf_max"
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
        raw_results.append(result)

    valid_indices = [index for index, item in enumerate(diagnostics) if item.valid]
    candidate_indices = valid_indices or [
        index for index, item in enumerate(diagnostics) if np.isfinite(item.objective)
    ]
    if not candidate_indices:
        raise RuntimeError("Every optimization start returned a non-finite objective.")
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

    payload = "|".join((str(master_seed), *(str(key) for key in keys))).encode("utf-8")
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
        seed = deterministic_seed(
            master_seed,
            "fit",
            participant,
            run_key,
            model_name.upper(),
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
