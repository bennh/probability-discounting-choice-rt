"""Competing value-to-reaction-time mappings with one shared interface."""

from __future__ import annotations

from typing import Any

import numpy as np


MODEL_NAMES = ("M1", "M2", "M3")


def _finite_scalar(name: str, value: Any, *, positive: bool = False) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(array)
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}.")
    return scalar


def _broadcast_trial_inputs(*values: Any) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value) for value in values)
    non_scalar_shapes = {array.shape for array in arrays if array.ndim > 0}
    if len(non_scalar_shapes) > 1:
        raise ValueError(
            "All non-scalar trial inputs must have exactly the same shape; "
            "only scalar inputs may be broadcast."
        )
    return tuple(np.broadcast_arrays(*arrays))


def rt_predictor(
    model_name: str,
    v_cert: Any,
    v_uncert: Any,
    delta_v: Any,
) -> np.ndarray:
    """Return the frozen predictor for M1, M2, or M3."""

    model = str(model_name).upper()
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown RT model {model_name!r}; expected one of {MODEL_NAMES}.")
    certain, uncertain, difference = _broadcast_trial_inputs(
        np.asarray(v_cert, dtype=float),
        np.asarray(v_uncert, dtype=float),
        np.asarray(delta_v, dtype=float),
    )
    if np.any(~np.isfinite(certain)) or np.any(~np.isfinite(uncertain)) or np.any(
        ~np.isfinite(difference)
    ):
        raise ValueError("RT predictor inputs must be finite.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            recomputed_difference = uncertain - certain
            if not np.allclose(
                difference,
                recomputed_difference,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError("delta_v must equal v_uncert - v_cert.")
            if model == "M1":
                predictor = np.abs(difference)
            elif model == "M2":
                predictor = np.square(difference)
            else:
                predictor = (np.abs(certain) + np.abs(uncertain)) / 2.0
        except FloatingPointError as exc:
            raise FloatingPointError(f"{model} predictor overflowed.") from exc
    if np.any(~np.isfinite(predictor)):
        raise FloatingPointError(f"{model} predictor produced non-finite values.")
    return predictor


def rt_location(
    alpha: float,
    delta_loss: float,
    b: float,
    condition: Any,
    predictor: Any,
) -> np.ndarray:
    """Compute log-RT location alpha + delta_L I(L) - b g(V)."""

    alpha_scalar = _finite_scalar("alpha", alpha)
    delta_loss_scalar = _finite_scalar("delta_loss", delta_loss)
    b_scalar = _finite_scalar("b", b, positive=True)
    labels, signal = _broadcast_trial_inputs(
        np.asarray(condition).astype(str), np.asarray(predictor, dtype=float)
    )
    if np.any(~np.isin(labels, ("R", "L"))):
        raise ValueError("RT condition labels must be R or L.")
    if np.any(~np.isfinite(signal)) or np.any(signal < 0.0):
        raise ValueError("RT predictors must be finite and non-negative.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            location = (
                alpha_scalar
                + delta_loss_scalar * (labels == "L")
                - b_scalar * signal
            )
        except FloatingPointError as exc:
            raise FloatingPointError("RT location exceeded the finite numeric range.") from exc
    if np.any(~np.isfinite(location)):
        raise FloatingPointError("RT location produced non-finite values.")
    return location
