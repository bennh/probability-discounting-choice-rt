#!/usr/bin/env python3
"""Fit the single choice-only baseline and all enabled full models on run A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import pandas as pd

from pd_project.config import load_config, repository_path, sha256_file
from pd_project.fit import (
    fit_dataset,
    fit_results_frame,
    start_diagnostics_frame,
)
from pd_project.provenance import (
    append_manifest,
    clean_git_commit,
    runtime_metadata,
    sha256_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    git_commit = clean_git_commit(config["_repository_root"])
    trials_path = repository_path(config, config["data"]["processed_trials"])
    trials = pd.read_csv(trials_path, dtype={"participant": "string"})
    run_a = trials.loc[trials["run"] == "A"].copy()
    if run_a.empty:
        raise RuntimeError("Processed data contains no run-A trials.")

    baseline = fit_dataset(
        run_a,
        "M1",
        config,
        choice_only=True,
    )
    full_results = []
    for model, specification in config["models"].items():
        if model.upper() in {"M1", "M2", "M3"} and specification.get("enabled") is True:
            full_results.extend(
                fit_dataset(
                    run_a,
                    model,
                    config,
                )
            )
    all_results = baseline + full_results
    output_dir = repository_path(config, "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_starts = int(config["optimization"]["multistarts"])
    config_hash = sha256_file(config["_config_path"])
    processed_hash = sha256_file(trials_path)
    runtime = runtime_metadata()
    runtime_hash = sha256_mapping(runtime)
    fits = fit_results_frame(all_results)
    fits["fit_run"] = "A"
    fits["multistarts_used"] = configured_starts
    fits["config_sha256"] = config_hash
    fits["processed_data_sha256"] = processed_hash
    fits["git_commit"] = git_commit
    fits["runtime_sha256"] = runtime_hash
    fits.to_csv(output_dir / "run_a_fits.csv", index=False)
    starts = start_diagnostics_frame(all_results)
    starts["fit_run"] = "A"
    starts["multistarts_used"] = configured_starts
    starts["config_sha256"] = config_hash
    starts["processed_data_sha256"] = processed_hash
    starts["git_commit"] = git_commit
    starts["runtime_sha256"] = runtime_hash
    starts.to_csv(output_dir / "run_a_optimizer_starts.csv", index=False)
    (output_dir / "run_a_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = sum(not result.success for result in all_results)
    timestamp = datetime.now(timezone.utc).isoformat()
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": "run_a_fits",
                "stage": "fit_run_a",
                "timestamp_utc": timestamp,
                "git_commit": git_commit,
                "config_sha256": config_hash,
                "processed_data_sha256": processed_hash,
                "artifact_sha256": sha256_file(output_dir / "run_a_fits.csv"),
                "run": "A",
                "fit_status": "passed" if not failures else "failures_present",
                "path": "results/run_a_fits.csv",
            },
            {
                "artifact": "run_a_optimizer_starts",
                "stage": "fit_run_a",
                "timestamp_utc": timestamp,
                "git_commit": git_commit,
                "config_sha256": config_hash,
                "processed_data_sha256": processed_hash,
                "artifact_sha256": sha256_file(
                    output_dir / "run_a_optimizer_starts.csv"
                ),
                "run": "A",
                "path": "results/run_a_optimizer_starts.csv",
            },
        ],
    )
    print(f"Saved {len(all_results)} run-A fits ({failures} flagged failures).")
    if failures:
        raise RuntimeError(
            "One or more run-A fits failed the frozen convergence checks; "
            "review diagnostics before continuing."
        )


if __name__ == "__main__":
    main()
