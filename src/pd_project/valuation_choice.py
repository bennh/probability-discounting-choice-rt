"""Fixed valuation model and logistic choice rule."""

from __future__ import annotations

from typing import NamedTuple, Any

import numpy as np


class SubjectiveValues(NamedTuple):
    v_cert: np.ndarray
    v_uncert: np.ndarray
    delta_v: np.ndarray


def parameter_by_condition(
    condition: Any, reward_value: float, loss_value: float
) -> np.ndarray:
    labels = np.asarray(condition).astype(str)
    valid = np.isin(labels, ("R", "L"))
    if np.any(~valid):
        raise ValueError(f"Unknown condition labels: {np.unique(labels[~valid]).tolist()}")
    reward = np.asarray(reward_value, dtype=float)
    loss = np.asarray(loss_value, dtype=float)
    if reward.ndim != 0 or loss.ndim != 0:
        raise ValueError("Condition-specific discount parameters must be scalars.")
    reward_scalar = float(reward)
    loss_scalar = float(loss)
    if (
        not np.isfinite(reward_scalar)
        or not np.isfinite(loss_scalar)
        or reward_scalar <= 0.0
        or loss_scalar <= 0.0
    ):
        raise ValueError(
            "Condition-specific discount parameters must be finite and positive."
        )
    return np.where(labels == "R", reward_scalar, loss_scalar)


def subjective_values(
    r_cert: Any,
    r_uncert: Any,
    odds: Any,
    k: Any,
    *,
    s0: float,
) -> SubjectiveValues:
    """Apply signed hyperbolic probability discounting."""

    if not np.isfinite(s0) or s0 <= 0.0:
        raise ValueError("s0 must be finite and positive.")
    inputs = (
        np.asarray(r_cert, dtype=float),
        np.asarray(r_uncert, dtype=float),
        np.asarray(odds, dtype=float),
        np.asarray(k, dtype=float),
    )
    non_scalar_shapes = {value.shape for value in inputs if value.ndim > 0}
    if len(non_scalar_shapes) > 1:
        raise ValueError(
            "All non-scalar trial inputs must have exactly the same shape; "
            "only scalar inputs may be broadcast."
        )
    certain, uncertain, odds_array, k_array = np.broadcast_arrays(*inputs)
    if np.any(~np.isfinite(certain)) or np.any(~np.isfinite(uncertain)):
        raise ValueError("Outcome amounts must be finite.")
    if np.any(~np.isfinite(odds_array)) or np.any(odds_array < 0.0):
        raise ValueError("Odds must be finite and non-negative.")
    if np.any(~np.isfinite(k_array)) or np.any(k_array <= 0.0):
        raise ValueError("Discount parameters must be finite and positive.")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            denominator = 1.0 + k_array * odds_array
            v_cert = certain / s0
            v_uncert = (uncertain / s0) / denominator
            delta_v = v_uncert - v_cert
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Subjective value computation exceeded the finite numeric range."
            ) from exc
    if (
        np.any(~np.isfinite(v_cert))
        or np.any(~np.isfinite(v_uncert))
        or np.any(~np.isfinite(delta_v))
    ):
        raise FloatingPointError("Subjective value computation produced non-finite values.")
    return SubjectiveValues(v_cert, v_uncert, delta_v)


def choice_logits(beta: float, delta_v: Any) -> np.ndarray:
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    logits = float(beta) * np.asarray(delta_v, dtype=float)
    if np.any(~np.isfinite(logits)):
        raise FloatingPointError("Choice logits are non-finite.")
    return logits


def choice_probability(logits: Any) -> np.ndarray:
    """Stable logistic transformation without first creating extreme odds."""

    values = np.asarray(logits, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Choice logits must be finite.")
    return np.exp(-np.logaddexp(0.0, -values))
