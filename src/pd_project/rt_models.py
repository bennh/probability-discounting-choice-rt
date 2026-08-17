"""Competing value-to-reaction-time mappings with one shared interface."""

from __future__ import annotations

from typing import Any

import numpy as np


MODEL_NAMES = ("M1", "M2", "M3")


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
    certain, uncertain, difference = np.broadcast_arrays(
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

    if not all(np.isfinite(value) for value in (alpha, delta_loss, b)) or b <= 0.0:
        raise ValueError("alpha/delta_loss must be finite and b must be finite/positive.")
    labels, signal = np.broadcast_arrays(
        np.asarray(condition).astype(str), np.asarray(predictor, dtype=float)
    )
    if np.any(~np.isin(labels, ("R", "L"))):
        raise ValueError("RT condition labels must be R or L.")
    if np.any(~np.isfinite(signal)):
        raise ValueError("RT predictors must be finite.")
    location = float(alpha) + float(delta_loss) * (labels == "L") - float(b) * signal
    if np.any(~np.isfinite(location)):
        raise FloatingPointError("RT location produced non-finite values.")
    return location

