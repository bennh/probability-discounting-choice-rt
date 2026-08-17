#!/usr/bin/env python3
"""Run a small simulate-refit smoke test before implementing formal recovery."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pd_project.config import load_config
from pd_project.data import odds_against
from pd_project.fit import fit_participant
from pd_project.recovery import simulate_participant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def synthetic_design() -> pd.DataFrame:
    probability = np.tile(np.array([0.10, 0.25, 0.50, 0.75, 0.90]), 8)
    n_trials = probability.size
    condition = np.where(np.arange(n_trials) % 2 == 0, "R", "L")
    signs = np.where(condition == "R", 1.0, -1.0)
    return pd.DataFrame(
        {
            "participant": "smoke_001",
            "run": "A",
            "trial_index": np.arange(1, n_trials + 1),
            "condition": condition,
            "r_cert": signs * 10.0,
            "r_uncert": signs * np.linspace(12.0, 30.0, n_trials),
            "probability": probability,
            "odds": odds_against(probability),
            "raw_action": np.ones(n_trials),
            "choice_uncertain": np.zeros(n_trials),
            "rt_seconds": np.ones(n_trials),
            "choice_included": True,
            "rt_included": True,
        }
    )


def main() -> None:
    args = parse_args()
    if not args.smoke:
        raise SystemExit(
            "Formal recovery orchestration must be frozen by the group first. "
            "Use --smoke to validate the simulate-refit kernel."
        )
    config = load_config(args.config)
    design = synthetic_design()
    parameters = {
        "log_k_R": np.log(1.0),
        "log_k_L": np.log(2.0),
        "log_beta": np.log(3.0),
        "alpha": np.log(2.0),
        "delta_L": 0.15,
        "log_b": np.log(0.4),
        "log_sigma": np.log(0.35),
    }
    for offset, model in enumerate(("M1", "M2", "M3")):
        simulated = simulate_participant(
            design,
            parameters,
            model,
            rng=np.random.default_rng(2026001 + offset),
            s0=10.0,
        )
        fitted = fit_participant(
            simulated,
            model,
            config,
            seed=2026101 + offset,
            multistarts=2,
        )
        if not np.isfinite(fitted.objective) or not fitted.success:
            raise RuntimeError(
                f"{model} smoke recovery failed: objective={fitted.objective}, "
                f"success={fitted.success}, message={fitted.message}"
            )
        print(f"{model}: objective={fitted.objective:.3f}, success={fitted.success}")


if __name__ == "__main__":
    main()
