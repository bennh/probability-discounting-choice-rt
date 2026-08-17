#!/usr/bin/env python3
"""Create compact participant-first result summaries for figures and report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from pd_project.config import load_config, repository_path, sha256_file
from pd_project.evaluation import participant_condition_means
from pd_project.provenance import append_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    results_dir = repository_path(config, "results/formal_run_b")
    scores = pd.read_csv(
        results_dir / "trial_scores.csv", dtype={"participant": "string"}
    )
    score_columns = [
        column
        for column in (
            "choice_log_score",
            "brier_score",
            "choice_correct",
            "rt_log_score",
            "absolute_log_rt_error",
            "absolute_rt_error_seconds",
        )
        if column in scores
    ]
    participant_condition = participant_condition_means(scores, score_columns)
    participant_condition.to_csv(results_dir / "participant_condition_scores.csv", index=False)
    group_summary = (
        participant_condition.groupby(["condition", "model"], as_index=False)[score_columns]
        .agg(["mean", "std", "count"])
    )
    group_summary.to_csv(results_dir / "group_score_summary.csv")
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": "participant_condition_scores",
                "stage": "make_outputs",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(config["_config_path"]),
                "artifact_sha256": sha256_file(
                    results_dir / "participant_condition_scores.csv"
                ),
                "path": "results/formal_run_b/participant_condition_scores.csv",
            },
            {
                "artifact": "group_score_summary",
                "stage": "make_outputs",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(config["_config_path"]),
                "artifact_sha256": sha256_file(
                    results_dir / "group_score_summary.csv"
                ),
                "path": "results/formal_run_b/group_score_summary.csv",
            },
        ],
    )
    print(f"Wrote summaries under {results_dir}")


if __name__ == "__main__":
    main()
