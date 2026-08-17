"""Stable choice, RT, and joint likelihood components."""

from __future__ import annotations

from typing import Any

import numpy as np


LOG_TWO_PI = float(np.log(2.0 * np.pi))


def bernoulli_logpmf_from_logits(choice: Any, logits: Any) -> np.ndarray:
    """Bernoulli log probability using logits, stable for extreme values."""

    y, z = np.broadcast_arrays(
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

    rt, location, scale = np.broadcast_arrays(
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
    log_rt = np.log(rt)
    standardized = (log_rt - location) / scale
    return -log_rt - np.log(scale) - 0.5 * LOG_TWO_PI - 0.5 * standardized**2


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
    include_choice = np.asarray(choice_mask, dtype=bool)
    include_rt = np.asarray(rt_mask, dtype=bool)
    shapes = {array.shape for array in (y, rt, z, location, include_choice, include_rt)}
    if len(shapes) != 1:
        raise ValueError("All trial-level arrays and masks must have matching shapes.")
    if np.any(include_rt & ~include_choice):
        raise ValueError("Every RT-likelihood trial must also have a valid choice.")
    choice_term = bernoulli_logpmf_from_logits(y[include_choice], z[include_choice]).sum()
    rt_term = lognormal_logpdf(
        rt[include_rt], location[include_rt], float(sigma)
    ).sum()
    total = float(choice_term + rt_term)
    if not np.isfinite(total):
        raise FloatingPointError("Joint log likelihood is non-finite.")
    return total

