#!/usr/bin/env python3
"""Fit the single choice-only baseline and all enabled full models on run A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from pd_project.config import load_config, repository_path, sha256_file
from pd_project.data import audit_trials, compute_s0
from pd_project.fit import (
    fit_dataset,
    fit_results_frame,
    start_diagnostics_frame,
    validate_run_a_fit_artifact,
)
from pd_project.provenance import (
    append_manifest,
    clean_git_commit,
    data_pipeline_source_hash,
    data_transform_contract_hash,
    raw_source_snapshot,
    runtime_metadata,
    sha256_mapping,
    sha256_named_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an earlier Run-A artifact set after explicit review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    git_commit = clean_git_commit(config["_repository_root"])
    data_config = config["data"]
    trials_path = repository_path(config, config["data"]["processed_trials"])
    audit_path = repository_path(config, data_config["audit_report"])
    if not audit_path.exists():
        raise RuntimeError("Run-A fitting requires a completed canonical data audit.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_passes = (
        "passed_integrity_checks",
        "passed_frozen_contract",
        "approved_for_fitting",
        "amount_scale_matches_expected",
    )
    failed_passes = [name for name in required_passes if audit.get(name) is not True]
    if failed_passes:
        raise RuntimeError(
            f"Run-A fitting is blocked by data-audit status: {failed_passes}."
        )

    processed_hash = sha256_file(trials_path)
    raw_files, current_archive_hash = raw_source_snapshot(
        config["_repository_root"],
        data_config,
        source_mode=str(audit.get("raw_source_mode", "")),
        expected_archive_sha256=audit.get("raw_archive_sha256"),
    )
    provenance_mismatches = {
        "processed_data_sha256": (
            audit.get("processed_data_sha256"),
            processed_hash,
        ),
        "raw_data_sha256": (
            audit.get("raw_data_sha256"),
            sha256_named_files(raw_files),
        ),
        "raw_archive_sha256": (
            audit.get("raw_archive_sha256"),
            current_archive_hash,
        ),
        "data_config_sha256": (
            audit.get("data_config_sha256"),
            data_transform_contract_hash(data_config),
        ),
        "data_pipeline_sha256": (
            audit.get("data_pipeline_sha256"),
            data_pipeline_source_hash(config["_repository_root"]),
        ),
    }
    provenance_mismatches = {
        name: values
        for name, values in provenance_mismatches.items()
        if values[0] != values[1]
    }
    if provenance_mismatches:
        raise RuntimeError(
            f"Run-A fitting is blocked by audit provenance mismatch: {provenance_mismatches}"
        )

    trials = pd.read_csv(trials_path, dtype={"participant": "string"})
    if sha256_file(trials_path) != processed_hash:
        raise RuntimeError("Processed data changed while Run-A fitting loaded it.")
    fresh_audit = audit_trials(trials, data_config)
    if fresh_audit["passed_frozen_contract"] is not True:
        raise RuntimeError(
            f"Processed data fail a fresh audit: {fresh_audit['deviations']}"
        )
    run_a = trials.loc[trials["run"] == "A"].copy()
    if run_a.empty:
        raise RuntimeError("Processed data contains no run-A trials.")
    scale = compute_s0(run_a["r_cert"].to_numpy(dtype=float))
    if not np.isclose(
        scale,
        float(data_config["transforms"]["amount_scale"]),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Processed data fail the frozen run-A amount-scale assertion.")

    participants = sorted(run_a["participant"].astype(str).unique())
    expected_participants = int(data_config["expected"]["participants"])
    if len(participants) != expected_participants:
        raise RuntimeError(
            f"Run A contains {len(participants)} participants; "
            f"the frozen contract requires {expected_participants}."
        )
    enabled_models = {
        str(model).upper()
        for model, specification in config["models"].items()
        if specification.get("enabled") is True
    }
    if enabled_models != {"M1", "M2", "M3"}:
        raise RuntimeError(
            "Run-A fitting requires exactly M1, M2, and M3 to be enabled; "
            f"found {sorted(enabled_models)}."
        )

    output_dir = repository_path(config, "results")
    receipt_path = repository_path(
        config, config["outputs"]["run_a_completion_receipt"]
    )
    final_paths = (
        output_dir / "run_a_fits.csv",
        output_dir / "run_a_optimizer_starts.csv",
        output_dir / "run_a_runtime.json",
        receipt_path,
    )
    existing = [str(path) for path in final_paths if path.exists()]
    if existing and not args.replace_existing:
        raise RuntimeError(
            "Run-A artifacts already exist; use --replace-existing only after review: "
            f"{existing}"
        )

    baseline = fit_dataset(
        run_a,
        "M1",
        config,
        choice_only=True,
    )
    full_results = []
    for model in ("M1", "M2", "M3"):
        full_results.extend(fit_dataset(run_a, model, config))
    all_results = baseline + full_results
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_starts = int(config["optimization"]["multistarts"])
    config_hash = sha256_file(config["_config_path"])
    runtime = runtime_metadata()
    runtime_hash = sha256_mapping(runtime)
    fits = fit_results_frame(all_results)
    expected_pairs = {
        (participant, model)
        for participant in participants
        for model in ("choice_only", "M1", "M2", "M3")
    }
    observed_pairs = [(str(result.participant), str(result.model)) for result in all_results]
    pair_counts = pd.Series(observed_pairs, dtype=object).value_counts()
    if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(expected_pairs):
        raise RuntimeError(
            "Fitting did not return the exact participant × choice_only/M1/M2/M3 matrix; "
            f"missing={sorted(expected_pairs - set(observed_pairs))}, "
            f"extra={sorted(set(observed_pairs) - expected_pairs)}, "
            f"duplicates={pair_counts.loc[pair_counts > 1].to_dict()}."
        )
    fits["fit_run"] = "A"
    fits["multistarts_used"] = configured_starts
    fits["config_sha256"] = config_hash
    fits["processed_data_sha256"] = processed_hash
    fits["data_pipeline_sha256"] = audit["data_pipeline_sha256"]
    fits["git_commit"] = git_commit
    fits["runtime_sha256"] = runtime_hash
    starts = start_diagnostics_frame(all_results)
    starts["fit_run"] = "A"
    starts["multistarts_used"] = configured_starts
    starts["config_sha256"] = config_hash
    starts["processed_data_sha256"] = processed_hash
    starts["data_pipeline_sha256"] = audit["data_pipeline_sha256"]
    starts["git_commit"] = git_commit
    starts["runtime_sha256"] = runtime_hash
    failures = sum(not result.success for result in all_results)
    if failures == 0:
        # This independently validates labels, booleans, objectives, parameter
        # columns, bounds, uniqueness, and the exact participant/model matrix.
        validate_run_a_fit_artifact(fits, participants, config)
    timestamp = datetime.now(timezone.utc).isoformat()
    # Fitting may take a long time. Recheck every mutable input immediately
    # before installing artifacts so the receipt cannot bind stale provenance.
    final_raw_files, final_archive_hash = raw_source_snapshot(
        config["_repository_root"], data_config,
        source_mode=str(audit["raw_source_mode"]),
        expected_archive_sha256=audit.get("raw_archive_sha256"),
    )
    final_checks = {
        "git_commit": (clean_git_commit(config["_repository_root"]), git_commit),
        "config_sha256": (sha256_file(config["_config_path"]), config_hash),
        "processed_data_sha256": (sha256_file(trials_path), processed_hash),
        "raw_data_sha256": (
            sha256_named_files(final_raw_files), audit["raw_data_sha256"]
        ),
        "raw_archive_sha256": (final_archive_hash, current_archive_hash),
        "data_pipeline_sha256": (
            data_pipeline_source_hash(config["_repository_root"]),
            audit["data_pipeline_sha256"],
        ),
    }
    changed = {name: pair for name, pair in final_checks.items() if pair[0] != pair[1]}
    if changed:
        raise RuntimeError(f"Run-A inputs changed during fitting: {changed}")
    with tempfile.TemporaryDirectory(prefix=".run_a_staging_", dir=output_dir) as name:
        staging = Path(name)
        staged_fits = staging / "run_a_fits.csv"
        staged_starts = staging / "run_a_optimizer_starts.csv"
        staged_runtime = staging / "run_a_runtime.json"
        staged_receipt = staging / receipt_path.name
        fits.to_csv(staged_fits, index=False)
        starts.to_csv(staged_starts, index=False)
        staged_runtime.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_hashes = {
            staged_fits.name: sha256_file(staged_fits),
            staged_starts.name: sha256_file(staged_starts),
            staged_runtime.name: sha256_file(staged_runtime),
        }
        receipt = {
            "status": "completed" if failures == 0 else "failed",
            "completed_at_utc": timestamp,
            "failures": failures,
            "fit_rows": len(all_results),
            "git_commit": git_commit,
            "config_sha256": config_hash,
            "raw_data_sha256": audit["raw_data_sha256"],
            "raw_archive_sha256": current_archive_hash,
            "raw_source_mode": audit["raw_source_mode"],
            "processed_data_sha256": processed_hash,
            "data_pipeline_sha256": audit["data_pipeline_sha256"],
            "runtime_sha256": runtime_hash,
            "multistarts_used": configured_starts,
            "artifacts": artifact_hashes,
        }
        staged_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staged_fits, output_dir / staged_fits.name)
        os.replace(staged_starts, output_dir / staged_starts.name)
        os.replace(staged_runtime, output_dir / staged_runtime.name)
        # The receipt is the commit marker and is always installed last.
        os.replace(staged_receipt, receipt_path)
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": "run_a_fits",
                "stage": "fit_run_a",
                "timestamp_utc": timestamp,
                "git_commit": git_commit,
                "config_sha256": config_hash,
                "raw_data_sha256": audit["raw_data_sha256"],
                "raw_archive_sha256": current_archive_hash or "",
                "raw_source_mode": audit["raw_source_mode"],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": audit["data_pipeline_sha256"],
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
                "raw_data_sha256": audit["raw_data_sha256"],
                "raw_archive_sha256": current_archive_hash or "",
                "raw_source_mode": audit["raw_source_mode"],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": audit["data_pipeline_sha256"],
                "artifact_sha256": sha256_file(
                    output_dir / "run_a_optimizer_starts.csv"
                ),
                "run": "A",
                "path": "results/run_a_optimizer_starts.csv",
            },
            {
                "artifact": "run_a_completion_receipt",
                "stage": "fit_run_a",
                "timestamp_utc": timestamp,
                "git_commit": git_commit,
                "config_sha256": config_hash,
                "raw_data_sha256": audit["raw_data_sha256"],
                "raw_archive_sha256": current_archive_hash or "",
                "raw_source_mode": audit["raw_source_mode"],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": audit["data_pipeline_sha256"],
                "artifact_sha256": sha256_file(receipt_path),
                "run": "A",
                "fit_status": "passed" if not failures else "failures_present",
                "path": str(receipt_path.relative_to(Path(config["_repository_root"]))),
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
