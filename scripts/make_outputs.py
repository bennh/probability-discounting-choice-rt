#!/usr/bin/env python3
"""Create verified participant-first summaries and frozen model contrasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from pd_project.config import load_config, repository_path, sha256_file
from pd_project.evaluation import holm_adjust, participant_condition_means
from pd_project.fit import deterministic_seed
from pd_project.provenance import append_manifest


SCORE_COLUMNS = (
    "choice_log_score", "brier_score", "choice_correct", "rt_log_score",
    "absolute_log_rt_error", "absolute_rt_error_seconds",
)
EXPECTED_MODELS = ("choice_only", "M1", "M2", "M3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--overwrite-derived", action="store_true",
        help="Replace only derived summaries, never formal Run-B inputs.",
    )
    return parser.parse_args()


def verify_formal_inputs(
    config: dict[str, Any], results_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Verify that scores belong to the completed one-shot Run-B execution."""

    score_path = results_dir / "trial_scores.csv"
    status_path = repository_path(config, config["run_b_guard"]["status_registry"])
    if not score_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("Completed formal Run-B scores and status are required.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError("Derived outputs require formal Run B to be completed.")
    expected_hash = status.get("output_sha256", {}).get("trial_scores.csv")
    if not expected_hash or sha256_file(score_path) != expected_hash:
        raise RuntimeError("trial_scores.csv does not match the completed Run-B record.")
    fingerprint = status.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("The Run-B completion record lacks an input fingerprint.")
    if fingerprint.get("config_sha256") != sha256_file(config["_config_path"]):
        raise RuntimeError("Current configuration differs from the Run-B configuration.")
    return score_path, status


def group_score_summary(participant_scores: pd.DataFrame) -> pd.DataFrame:
    """Produce a flat long table with equal participant weighting."""

    metrics = [column for column in SCORE_COLUMNS if column in participant_scores]
    long = participant_scores.melt(
        id_vars=["participant", "condition", "model"], value_vars=metrics,
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    if long.empty:
        raise ValueError("No participant-level scores are available to summarize.")
    return (
        long.groupby(["condition", "model", "metric"], as_index=False, sort=True)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    )


def _paired_contrast(
    participant_scores: pd.DataFrame,
    *,
    condition: str,
    metric: str,
    model_1: str,
    model_2: str,
    n_boot: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    subset = participant_scores.loc[
        (participant_scores["condition"] == condition)
        & participant_scores["model"].isin((model_1, model_2)),
        ["participant", "model", metric],
    ]
    if subset.duplicated(["participant", "model"]).any():
        raise ValueError("Contrasts require one row per participant and model.")
    paired = subset.pivot(index="participant", columns="model", values=metric)
    if model_1 not in paired or model_2 not in paired:
        raise ValueError(f"Missing model in contrast {model_1} versus {model_2}.")
    complete = paired[[model_1, model_2]].dropna()
    if len(complete) < 2:
        raise ValueError(f"Too few complete pairs for {model_1} versus {model_2}.")
    difference = (complete[model_1] - complete[model_2]).to_numpy(dtype=float)
    if np.any(~np.isfinite(difference)):
        raise ValueError("Paired model differences must be finite.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(n_boot, difference.size))
    bootstrap = difference[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    observed = float(difference.mean())
    null_bootstrap = (difference - observed)[indices].mean(axis=1)
    # Centering imposes the null mean of zero; the add-one correction avoids
    # reporting an impossible exact p=0 from a finite Monte Carlo sample.
    p_value = (1 + np.count_nonzero(np.abs(null_bootstrap) >= abs(observed))) / (
        n_boot + 1
    )
    return {
        "condition": condition,
        "metric": metric,
        "model_1": model_1,
        "model_2": model_2,
        "difference_model_1_minus_model_2": observed,
        "ci_low": float(np.quantile(bootstrap, alpha / 2.0)),
        "ci_high": float(np.quantile(bootstrap, 1.0 - alpha / 2.0)),
        "p_value_two_sided": float(p_value),
        "n_participants": int(difference.size),
        "bootstrap_seed": int(seed),
    }


def model_comparisons(
    participant_scores: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Compute the two predeclared comparison families with Holm correction."""

    bootstrap_config = config["bootstrap"]
    n_boot = bootstrap_config["resamples"]
    confidence_level = float(bootstrap_config["confidence_level"])
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("bootstrap.resamples must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap.confidence_level must lie inside (0, 1).")
    master_seed = int(config["random"]["master_seed"])
    families = {
        "rt_models": (
            "rt_log_score", (("M1", "M2"), ("M1", "M3"), ("M2", "M3")),
        ),
        "rt_informed_vs_choice_only": (
            "choice_log_score",
            (("M1", "choice_only"), ("M2", "choice_only"),
             ("M3", "choice_only")),
        ),
    }
    rows: list[dict[str, Any]] = []
    for family, (metric, contrasts) in families.items():
        for condition in ("R", "L"):
            for model_1, model_2 in contrasts:
                seed = deterministic_seed(
                    master_seed, "formal_comparison", family, condition,
                    model_1, model_2,
                )
                rows.append(
                    {
                        "family": family,
                        **_paired_contrast(
                            participant_scores, condition=condition, metric=metric,
                            model_1=model_1, model_2=model_2, n_boot=n_boot,
                            confidence_level=confidence_level, seed=seed,
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    output["p_value_holm"] = np.nan
    for _, indices in output.groupby("family", sort=False).groups.items():
        output.loc[indices, "p_value_holm"] = holm_adjust(
            output.loc[indices, "p_value_two_sided"].to_numpy(dtype=float)
        )
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(config["_repository_root"])
    results_dir = repository_path(config, "results/formal_run_b")
    score_path, status = verify_formal_inputs(config, results_dir)
    scores = pd.read_csv(score_path, dtype={"participant": "string"})
    missing = sorted({"participant", "condition", "model"} - set(scores.columns))
    if missing:
        raise ValueError(f"Trial-score table is missing columns: {missing}")
    observed_models = set(scores["model"].dropna().astype(str))
    if observed_models != set(EXPECTED_MODELS):
        raise ValueError(f"Unexpected trial-score model set: {sorted(observed_models)}")
    score_columns = [column for column in SCORE_COLUMNS if column in scores]
    if not score_columns:
        raise ValueError("Trial-score table contains no recognized score columns.")

    participant_condition = participant_condition_means(scores, score_columns)
    outputs = {
        "participant_condition_scores.csv": participant_condition,
        "group_score_summary.csv": group_score_summary(participant_condition),
        "model_comparisons.csv": model_comparisons(participant_condition, config),
    }
    output_paths = [results_dir / filename for filename in outputs]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite_derived:
        raise RuntimeError(
            "Derived outputs already exist; use --overwrite-derived after verification: "
            f"{existing}"
        )
    with tempfile.TemporaryDirectory(prefix=".summary_staging_", dir=results_dir) as name:
        staging = Path(name)
        for filename, frame in outputs.items():
            frame.to_csv(staging / filename, index=False)
        for filename in outputs:
            os.replace(staging / filename, results_dir / filename)

    timestamp = datetime.now(timezone.utc).isoformat()
    fingerprint = status["fingerprint"]
    records = []
    for filename in outputs:
        path = results_dir / filename
        records.append(
            {
                "artifact": Path(filename).stem,
                "stage": "make_outputs",
                "timestamp_utc": timestamp,
                "git_commit": fingerprint.get("git_commit", ""),
                "config_sha256": fingerprint["config_sha256"],
                "raw_data_sha256": fingerprint.get("raw_data_sha256", ""),
                "raw_archive_sha256": fingerprint.get("raw_archive_sha256", ""),
                "raw_source_mode": fingerprint.get("raw_source_mode", ""),
                "processed_data_sha256": fingerprint.get("processed_data_sha256", ""),
                "data_pipeline_sha256": fingerprint.get("data_pipeline_sha256", ""),
                "artifact_sha256": sha256_file(path),
                "run": "B", "fit_status": "passed",
                "path": str(path.relative_to(root)),
            }
        )
    append_manifest(repository_path(config, config["outputs"]["manifest"]), records)
    print(f"Wrote verified participant summaries and comparisons under {results_dir}")


if __name__ == "__main__":
    main()
