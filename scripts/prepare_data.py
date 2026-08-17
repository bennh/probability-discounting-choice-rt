#!/usr/bin/env python3
"""Build and audit the canonical tidy PD dataset from local MATLAB files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from pd_project.config import load_config, repository_path
from pd_project.data import audit_trials, compute_s0, load_data_files
from pd_project.config import sha256_file
from pd_project.provenance import (
    append_manifest,
    data_pipeline_source_hash,
    data_transform_contract_hash,
    materialize_archive_generation,
    raw_source_snapshot,
    sha256_named_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--allow-count-deviations",
        action="store_true",
        help=(
            "Write a diagnostic-only dataset when counts differ; integrity failures "
            "are never bypassed and diagnostic data cannot be fitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    raw_directory = repository_path(config, data_config["raw_directory"])
    archive_path = repository_path(config, data_config["raw_archive"])
    raw_glob = data_config.get("raw_glob", "*.mat")
    if archive_path.is_file():
        raw_source_mode = "archive"
        raw_archive_hash: str | None = sha256_file(archive_path)
        source_directory = raw_directory / "extracted" / raw_archive_hash
        materialize_archive_generation(
            archive_path,
            source_directory,
            archive_hash=raw_archive_hash,
            raw_glob=raw_glob,
        )
    else:
        raw_source_mode = "direct_mat"
        raw_archive_hash = None
        source_directory = raw_directory
    raw_files, _ = raw_source_snapshot(
        config["_repository_root"],
        data_config,
        source_mode=raw_source_mode,
        expected_archive_sha256=raw_archive_hash,
    )
    raw_hash_before_load = sha256_named_files(raw_files)
    trials = load_data_files(raw_files, data_config)
    raw_hash_after_load = sha256_named_files(raw_files)
    if raw_hash_after_load != raw_hash_before_load:
        raise RuntimeError("Raw MATLAB files changed while they were being loaded.")

    run_a = trials.loc[trials["run"] == "A"]
    scale = compute_s0(run_a["r_cert"].to_numpy())
    expected_scale = float(data_config["transforms"]["amount_scale"])
    audit = audit_trials(trials, data_config)
    audit["amount_scale_s0"] = scale
    audit["amount_scale_matches_expected"] = bool(
        np.isclose(scale, expected_scale, atol=1.0e-12, rtol=0.0)
    )
    valid_run_a_rt = run_a.loc[run_a["rt_included"], "rt_seconds"].to_numpy(dtype=float)
    if valid_run_a_rt.size == 0:
        raise RuntimeError("No valid run-A RT values are available for sensitivity cutoffs.")
    lower_q, upper_q = data_config["rt_sensitivity"]["quantiles_from_run_a"]
    audit["run_a_rt_sensitivity_seconds"] = [
        float(np.quantile(valid_run_a_rt, lower_q)),
        float(np.quantile(valid_run_a_rt, upper_q)),
    ]

    if audit["integrity_deviations"]:
        raise RuntimeError(
            "Dataset integrity checks failed and cannot be bypassed: "
            f"{audit['integrity_deviations']}"
        )
    if audit["count_deviations"] and not args.allow_count_deviations:
        raise RuntimeError(
            f"Dataset counts differ from the candidate contract: {audit['count_deviations']}. "
            "Resolve the discrepancy or use --allow-count-deviations to write a "
            "diagnostic-only artifact."
        )
    if not audit["amount_scale_matches_expected"]:
        raise RuntimeError(
            f"Observed s0={scale} differs from configured value {expected_scale}; resolve before fitting."
        )

    diagnostic_only = bool(audit["count_deviations"])
    audit["diagnostic_only"] = diagnostic_only
    audit["approved_for_fitting"] = bool(
        audit["passed_frozen_contract"]
        and audit["amount_scale_matches_expected"]
        and not diagnostic_only
    )

    canonical_output_path = repository_path(config, data_config["processed_trials"])
    canonical_audit_path = repository_path(config, data_config["audit_report"])
    if diagnostic_only:
        output_path = canonical_output_path.with_name(
            f"{canonical_output_path.stem}.diagnostic{canonical_output_path.suffix}"
        )
        audit_path = canonical_audit_path.with_name(
            f"{canonical_audit_path.stem}.diagnostic{canonical_audit_path.suffix}"
        )
    else:
        output_path = canonical_output_path
        audit_path = canonical_audit_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_path, index=False)
    audit["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["raw_data_sha256"] = raw_hash_after_load
    audit["raw_source_mode"] = raw_source_mode
    audit["raw_archive_sha256"] = raw_archive_hash
    audit["processed_data_sha256"] = sha256_file(output_path)
    audit["data_config_sha256"] = data_transform_contract_hash(data_config)
    audit["data_pipeline_sha256"] = data_pipeline_source_hash(
        config["_repository_root"]
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": (
                    "processed_trials_diagnostic" if diagnostic_only else "processed_trials"
                ),
                "stage": "prepare_data",
                "timestamp_utc": audit["prepared_at_utc"],
                "config_sha256": sha256_file(config["_config_path"]),
                "raw_data_sha256": audit["raw_data_sha256"],
                "raw_source_mode": raw_source_mode,
                "raw_archive_sha256": raw_archive_hash or "",
                "processed_data_sha256": audit["processed_data_sha256"],
                "data_pipeline_sha256": audit["data_pipeline_sha256"],
                "artifact_sha256": audit["processed_data_sha256"],
                "path": str(output_path.relative_to(Path(config["_repository_root"]))),
            },
            {
                "artifact": "data_audit",
                "stage": "prepare_data",
                "timestamp_utc": audit["prepared_at_utc"],
                "config_sha256": sha256_file(config["_config_path"]),
                "raw_data_sha256": audit["raw_data_sha256"],
                "raw_source_mode": raw_source_mode,
                "raw_archive_sha256": raw_archive_hash or "",
                "processed_data_sha256": audit["processed_data_sha256"],
                "data_pipeline_sha256": audit["data_pipeline_sha256"],
                "artifact_sha256": sha256_file(audit_path),
                "fit_status": "passed" if audit["approved_for_fitting"] else "diagnostic_only",
                "path": str(audit_path.relative_to(Path(config["_repository_root"]))),
            },
        ],
    )
    print(f"Wrote {len(trials):,} trials to {output_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
