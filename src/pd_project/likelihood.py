"""Stable choice, RT, and joint likelihood components."""

from __future__ import annotations

from typing import Any

import numpy as np


LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _broadcast_likelihood_inputs(*values: Any) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value) for value in values)
    non_scalar_shapes = {array.shape for array in arrays if array.ndim > 0}
    if len(non_scalar_shapes) > 1:
        raise ValueError(
            "All non-scalar likelihood inputs must have exactly the same shape; "
            "only scalar inputs may be broadcast."
        )
    return tuple(np.broadcast_arrays(*arrays))


def _strict_boolean_mask(name: str, value: Any) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype.")
    return mask.astype(bool, copy=False)


def _positive_finite_scalar(name: str, value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def bernoulli_logpmf_from_logits(choice: Any, logits: Any) -> np.ndarray:
    """Bernoulli log probability using logits, stable for extreme values."""

    y, z = _broadcast_likelihood_inputs(
        np.asarray(choice, dtype=float), np.asarray(logits, dtype=float)
    )
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(z)):
        raise ValueError("Choices and selected logits must be finite.")
    if np.any(~np.isin(y, (0.0, 1.0))):
        raise ValueError("Choices must be coded 0 or 1.")
    signed_error = np.where(y == 1.0, -z, z)
    return -np.logaddexp(0.0, signed_error)


def lognormal_logpdf(rt_seconds: Any, mu: Any, sigma: Any) -> np.ndarray:
    """Proper seconds-scale log-normal density, including -log(RT)."""

    rt, location, scale = _broadcast_likelihood_inputs(
        np.asarray(rt_seconds, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if np.any(~np.isfinite(rt)) or np.any(rt <= 0.0):
        raise ValueError("Included RT values must be finite and positive.")
    if np.any(~np.isfinite(location)):
        raise ValueError("RT locations must be finite.")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("RT sigma must be finite and positive.")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            log_rt = np.log(rt)
            standardized = (log_rt - location) / scale
            log_density = (
                -log_rt
                - np.log(scale)
                - 0.5 * LOG_TWO_PI
                - 0.5 * np.square(standardized)
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Log-normal density exceeded the finite numeric range."
            ) from exc
    if np.any(~np.isfinite(log_density)):
        raise FloatingPointError("Log-normal density produced non-finite values.")
    return log_density


def joint_log_likelihood(
    choice_uncertain: Any,
    rt_seconds: Any,
    logits: Any,
    mu: Any,
    sigma: float,
    *,
    choice_mask: Any,
    rt_mask: Any,
) -> float:
    """Sum choice and RT terms using their separately supplied masks."""

    y = np.asarray(choice_uncertain, dtype=float)
    rt = np.asarray(rt_seconds, dtype=float)
    z = np.asarray(logits, dtype=float)
    location = np.asarray(mu, dtype=float)
    include_choice = _strict_boolean_mask("choice_mask", choice_mask)
    include_rt = _strict_boolean_mask("rt_mask", rt_mask)
    sigma_scalar = _positive_finite_scalar("sigma", sigma)
    shapes = {array.shape for array in (y, rt, z, location, include_choice, include_rt)}
    if len(shapes) != 1:
        raise ValueError("All trial-level arrays and masks must have matching shapes.")
    if np.any(include_rt & ~include_choice):
        raise ValueError("Every RT-likelihood trial must also have a valid choice.")
    choice_term = bernoulli_logpmf_from_logits(y[include_choice], z[include_choice]).sum()
    rt_term = lognormal_logpdf(
        rt[include_rt], location[include_rt], sigma_scalar
    ).sum()
    total = float(choice_term + rt_term)
    if not np.isfinite(total):
        raise FloatingPointError("Joint log likelihood is non-finite.")
    return total
