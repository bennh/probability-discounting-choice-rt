#!/usr/bin/env python3
"""Build and audit the canonical tidy PD dataset from local MATLAB files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import zipfile

import numpy as np

from pd_project.config import load_config, repository_path
from pd_project.data import audit_trials, compute_s0, load_data_directory
from pd_project.config import sha256_file
from pd_project.provenance import (
    append_manifest,
    data_transform_contract_hash,
    sha256_named_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--allow-count-deviations",
        action="store_true",
        help="Write processed data despite count deviations; record them in the audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    raw_directory = repository_path(config, data_config["raw_directory"])
    raw_files = sorted(raw_directory.glob(data_config.get("raw_glob", "*.mat")))
    if not raw_files:
        archive_path = repository_path(config, data_config["raw_archive"])
        if not archive_path.exists():
            raise FileNotFoundError(
                f"No .mat files or raw archive found. Expected {archive_path}."
            )
        extraction_directory = (raw_directory / "extracted").resolve()
        extraction_directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (extraction_directory / member.filename).resolve()
                if not target.is_relative_to(extraction_directory):
                    raise RuntimeError(f"Unsafe path in raw archive: {member.filename}")
            archive.extractall(extraction_directory)
        raw_files = sorted(raw_directory.glob(data_config.get("raw_glob", "*.mat")))
        if not raw_files:
            raise FileNotFoundError("The raw archive contained no MATLAB participant files.")
    trials = load_data_directory(raw_directory, data_config)

    run_a = trials.loc[trials["run"] == "A"]
    scale = compute_s0(run_a["r_cert"].to_numpy())
    expected_scale = float(data_config["transforms"]["amount_scale"])
    audit = audit_trials(trials, data_config)
    audit["amount_scale_s0"] = scale
    audit["amount_scale_matches_expected"] = bool(np.isclose(scale, expected_scale))
    valid_run_a_rt = run_a.loc[run_a["rt_included"], "rt_seconds"].to_numpy(dtype=float)
    lower_q, upper_q = data_config["rt_sensitivity"]["quantiles_from_run_a"]
    audit["run_a_rt_sensitivity_seconds"] = [
        float(np.quantile(valid_run_a_rt, lower_q)),
        float(np.quantile(valid_run_a_rt, upper_q)),
    ]

    if audit["deviations"] and not args.allow_count_deviations:
        raise RuntimeError(
            f"Dataset counts differ from the candidate contract: {audit['deviations']}. "
            "Resolve the discrepancy or rerun with --allow-count-deviations and log the decision."
        )
    if not audit["amount_scale_matches_expected"]:
        raise RuntimeError(
            f"Observed s0={scale} differs from configured value {expected_scale}; resolve before fitting."
        )

    output_path = repository_path(config, data_config["processed_trials"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_path, index=False)
    audit["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["raw_data_sha256"] = sha256_named_files(raw_files)
    audit["processed_data_sha256"] = sha256_file(output_path)
    audit["data_config_sha256"] = data_transform_contract_hash(data_config)
    audit_path = repository_path(config, data_config["audit_report"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": "processed_trials",
                "stage": "prepare_data",
                "timestamp_utc": audit["prepared_at_utc"],
                "config_sha256": sha256_file(config["_config_path"]),
                "raw_data_sha256": audit["raw_data_sha256"],
                "processed_data_sha256": audit["processed_data_sha256"],
                "artifact_sha256": audit["processed_data_sha256"],
                "path": str(output_path.relative_to(Path(config["_repository_root"]))),
            },
            {
                "artifact": "data_audit",
                "stage": "prepare_data",
                "timestamp_utc": audit["prepared_at_utc"],
                "config_sha256": sha256_file(config["_config_path"]),
                "raw_data_sha256": audit["raw_data_sha256"],
                "processed_data_sha256": audit["processed_data_sha256"],
                "artifact_sha256": sha256_file(audit_path),
                "fit_status": "passed" if audit["passed_expected_counts"] else "deviation",
                "path": str(audit_path.relative_to(Path(config["_repository_root"]))),
            },
        ],
    )
    print(f"Wrote {len(trials):,} trials to {output_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
