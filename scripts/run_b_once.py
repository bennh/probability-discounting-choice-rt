#!/usr/bin/env python3
"""One-shot held-out scoring and independently initialized run-B reliability fits."""

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

from pd_project.config import (
    assert_formal_run_b_authorized,
    load_config,
    repository_path,
    sha256_file,
)
from pd_project.evaluation import support_shift_flags, trial_scores
from pd_project.data import audit_trials, compute_s0
from pd_project.fit import (
    FULL_PARAMETER_NAMES,
    CHOICE_PARAMETER_NAMES,
    fit_dataset,
    fit_results_frame,
    predict_from_estimates,
    start_diagnostics_frame,
    validate_run_a_fit_artifact,
)
from pd_project.likelihood import bernoulli_logpmf_from_logits
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
from pd_project.valuation_choice import choice_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    return parser.parse_args()


def parameter_dict(row: pd.Series, names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(row[name]) for name in names}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Install a state record atomically so interruption cannot truncate it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_score_artifact(
    scores: pd.DataFrame,
    run_b: pd.DataFrame,
    participants: list[str],
) -> None:
    """Validate the exact held-out participant/model/trial score matrix."""

    required = {
        "model",
        "participant",
        "condition",
        "trial_index",
        "choice_log_score",
        "brier_score",
        "choice_correct",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise RuntimeError(f"Held-out score artifact lacks columns: {missing}")
    if scores[["model", "participant", "condition", "trial_index"]].isna().any().any():
        raise RuntimeError("Held-out score keys must not be missing.")
    expected_models = {"choice_only", "M1", "M2", "M3"}
    if set(scores["model"].astype(str)) != expected_models:
        raise RuntimeError("Held-out score artifact has an invalid model set.")
    expected_trials = run_b.groupby("participant", sort=True).size()
    observed = scores.groupby(["participant", "model"], sort=True).size()
    expected_index = pd.MultiIndex.from_product(
        [participants, sorted(expected_models)], names=["participant", "model"]
    )
    if set(observed.index) != set(expected_index):
        raise RuntimeError(
            "Held-out score artifact has an incomplete participant/model matrix."
        )
    for (participant, _), count in observed.items():
        if int(count) != int(expected_trials.loc[participant]):
            raise RuntimeError("Held-out score rows do not match Run-B trial counts.")
    if scores.duplicated(["participant", "model", "trial_index"]).any():
        raise RuntimeError("Held-out score trial keys are not unique.")
    source_masks = run_b.loc[
        :,
        [
            "participant",
            "trial_index",
            "condition",
            "choice_included",
            "rt_included",
        ],
    ].copy()
    source_masks["participant"] = source_masks["participant"].astype(str)
    checked = scores.merge(
        source_masks,
        on=["participant", "trial_index"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_source"),
    )
    if (
        checked[["choice_included", "rt_included", "condition_source"]]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError("Held-out score keys do not map completely to Run-B trials.")
    if not checked["condition"].astype(str).eq(
        checked["condition_source"].astype(str)
    ).all():
        raise RuntimeError("Held-out score conditions disagree with Run-B trials.")
    choice_finite = np.isfinite(
        pd.to_numeric(
            checked["choice_log_score"], errors="coerce"
        ).to_numpy(float)
    )
    if not np.array_equal(
        choice_finite, checked["choice_included"].to_numpy(bool)
    ):
        raise RuntimeError("Choice-score missingness disagrees with the choice mask.")
    full = scores["model"].astype(str).ne("choice_only")
    required_full = {
        "rt_log_score",
        "absolute_log_rt_error",
        "absolute_rt_error_seconds",
        "out_of_support",
    }
    if required_full - set(scores.columns):
        raise RuntimeError("Full-model score rows lack RT/support diagnostics.")
    if scores.loc[full, "out_of_support"].isna().any():
        raise RuntimeError("Full-model support flags must be complete.")
    full_checked = checked.loc[
        checked["model"].astype(str).ne("choice_only")
    ]
    rt_finite = np.isfinite(
        pd.to_numeric(
            full_checked["rt_log_score"], errors="coerce"
        ).to_numpy(float)
    )
    if not np.array_equal(
        rt_finite, full_checked["rt_included"].to_numpy(bool)
    ):
        raise RuntimeError("RT-score missingness disagrees with the RT mask.")
    baseline = scores["model"].astype(str).eq("choice_only")
    if (
        scores.loc[
            baseline,
            [
                "rt_log_score",
                "absolute_log_rt_error",
                "absolute_rt_error_seconds",
            ],
        ]
        .notna()
        .any()
        .any()
    ):
        raise RuntimeError("Choice-only rows must not contain RT scores.")
    numeric_columns = [
        "choice_log_score",
        "brier_score",
        "choice_correct",
        "rt_log_score",
        "absolute_log_rt_error",
        "absolute_rt_error_seconds",
    ]
    numeric = scores[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise RuntimeError("Held-out scores contain infinity.")


def validate_reliability_matrix(
    results: list[Any], participants: list[str], config: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    """Validate reliability fit coverage and all successful fit artifacts."""

    frame = fit_results_frame(results)
    expected = {
        (participant, model)
        for participant in participants
        for model in ("choice_only", "M1", "M2", "M3")
    }
    observed = [
        (str(result.participant), str(result.model)) for result in results
    ]
    if set(observed) != expected or len(observed) != len(expected):
        raise RuntimeError(
            "Run-B reliability fits do not form the exact participant/model matrix."
        )
    failures = sum(not result.success for result in results)
    if failures == 0:
        validate_run_a_fit_artifact(frame, participants, config)
    return frame, failures


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_formal_run_b_authorized(config)

    root = Path(config["_repository_root"])
    lock_directory = root / ".formal_run_b"
    status_path = repository_path(
        config, config["run_b_guard"]["status_registry"]
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "not_run":
        raise RuntimeError(
            "The tracked formal run-B registry is not 'not_run'. Refusing to start."
        )

    commit = clean_git_commit(root)
    config_hash = sha256_file(config["_config_path"])
    processed_path = repository_path(
        config, config["data"]["processed_trials"]
    )
    run_a_fits_path = repository_path(config, "results/run_a_fits.csv")
    run_a_starts_path = repository_path(
        config, "results/run_a_optimizer_starts.csv"
    )
    run_a_runtime_path = repository_path(
        config, "results/run_a_runtime.json"
    )
    run_a_receipt_path = repository_path(
        config, config["outputs"]["run_a_completion_receipt"]
    )
    audit_path = repository_path(
        config, config["data"]["audit_report"]
    )
    audit_hash = sha256_file(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    raw_files, current_archive_hash = raw_source_snapshot(
        root,
        config["data"],
        source_mode=str(audit.get("raw_source_mode", "")),
        expected_archive_sha256=audit.get("raw_archive_sha256"),
    )
    raw_hash = sha256_named_files(raw_files)

    required_audit_passes = (
        "passed_integrity_checks",
        "passed_frozen_contract",
        "approved_for_fitting",
        "amount_scale_matches_expected",
    )
    failed_audit_passes = [
        name
        for name in required_audit_passes
        if audit.get(name) is not True
    ]
    if failed_audit_passes:
        raise RuntimeError(
            "Formal run B requires a canonical data audit; failed fields: "
            f"{failed_audit_passes}."
        )

    resolved_rt_cutoffs = config["data"]["rt_sensitivity"].get(
        "resolved_seconds"
    )
    if (
        not isinstance(resolved_rt_cutoffs, list)
        or len(resolved_rt_cutoffs) != 2
    ):
        raise RuntimeError(
            "Formal run B requires two frozen resolved RT sensitivity cutoffs."
        )

    audited_rt_cutoffs = audit.get("run_a_rt_sensitivity_seconds")
    if (
        not isinstance(audited_rt_cutoffs, list)
        or len(audited_rt_cutoffs) != 2
        or not np.allclose(
            np.asarray(resolved_rt_cutoffs, dtype=float),
            np.asarray(audited_rt_cutoffs, dtype=float),
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "Frozen RT sensitivity cutoffs do not match the run-A audit."
        )

    processed_hash = sha256_file(processed_path)
    data_config_hash = data_transform_contract_hash(config["data"])
    data_pipeline_hash = data_pipeline_source_hash(root)

    provenance_mismatches = {
        "raw_data_sha256": (
            audit.get("raw_data_sha256"),
            raw_hash,
        ),
        "raw_archive_sha256": (
            audit.get("raw_archive_sha256"),
            current_archive_hash,
        ),
        "processed_data_sha256": (
            audit.get("processed_data_sha256"),
            processed_hash,
        ),
        "data_config_sha256": (
            audit.get("data_config_sha256"),
            data_config_hash,
        ),
        "data_pipeline_sha256": (
            audit.get("data_pipeline_sha256"),
            data_pipeline_hash,
        ),
    }
    provenance_mismatches = {
        name: values
        for name, values in provenance_mismatches.items()
        if values[0] != values[1]
    }
    if provenance_mismatches:
        raise RuntimeError(
            "Data audit provenance does not match current inputs: "
            f"{provenance_mismatches}"
        )

    if not run_a_receipt_path.is_file():
        raise RuntimeError(
            "Formal run B requires a completed Run-A receipt."
        )

    run_a_receipt_hash = sha256_file(run_a_receipt_path)
    run_a_receipt = json.loads(
        run_a_receipt_path.read_text(encoding="utf-8")
    )

    run_a_config_hash = run_a_receipt.get("config_sha256")
    run_a_commit = run_a_receipt.get("git_commit")
    if not isinstance(run_a_config_hash, str) or not run_a_config_hash:
        raise RuntimeError(
            "Run-A completion receipt lacks a valid config_sha256."
        )
    if not isinstance(run_a_commit, str) or not run_a_commit:
        raise RuntimeError(
            "Run-A completion receipt lacks a valid git_commit."
        )

    current_runtime_hash = sha256_mapping(runtime_metadata())
    run_a_artifact_hashes = {
        "run_a_fits.csv": sha256_file(run_a_fits_path),
        "run_a_optimizer_starts.csv": sha256_file(run_a_starts_path),
        "run_a_runtime.json": sha256_file(run_a_runtime_path),
    }

    receipt_mismatches = {
        "status": (
            run_a_receipt.get("status"),
            "completed",
        ),
        "failures": (
            run_a_receipt.get("failures"),
            0,
        ),
        "raw_data_sha256": (
            run_a_receipt.get("raw_data_sha256"),
            raw_hash,
        ),
        "raw_archive_sha256": (
            run_a_receipt.get("raw_archive_sha256"),
            current_archive_hash,
        ),
        "raw_source_mode": (
            run_a_receipt.get("raw_source_mode"),
            audit["raw_source_mode"],
        ),
        "processed_data_sha256": (
            run_a_receipt.get("processed_data_sha256"),
            processed_hash,
        ),
        "data_pipeline_sha256": (
            run_a_receipt.get("data_pipeline_sha256"),
            data_pipeline_hash,
        ),
        "runtime_sha256": (
            run_a_receipt.get("runtime_sha256"),
            current_runtime_hash,
        ),
        "multistarts_used": (
            run_a_receipt.get("multistarts_used"),
            int(config["optimization"]["multistarts"]),
        ),
        "artifacts": (
            run_a_receipt.get("artifacts"),
            run_a_artifact_hashes,
        ),
    }
    receipt_mismatches = {
        name: values
        for name, values in receipt_mismatches.items()
        if values[0] != values[1]
    }
    if receipt_mismatches:
        raise RuntimeError(
            "Run-A completion receipt does not match current artifacts: "
            f"{receipt_mismatches}"
        )

    fingerprint = {
        "config_sha256": config_hash,
        "git_commit": commit,
        "raw_data_sha256": raw_hash,
        "raw_archive_sha256": current_archive_hash,
        "raw_source_mode": audit["raw_source_mode"],
        "processed_data_sha256": processed_hash,
        "data_pipeline_sha256": data_pipeline_hash,
        "run_a_fits_sha256": run_a_artifact_hashes["run_a_fits.csv"],
        "run_a_receipt_sha256": run_a_receipt_hash,
        "runtime_sha256": current_runtime_hash,
    }

    trials = pd.read_csv(
        processed_path,
        dtype={"participant": "string"},
    )
    run_a = trials.loc[trials["run"] == "A"].copy()
    run_b = trials.loc[trials["run"] == "B"].copy()

    recomputed_audit = audit_trials(trials, config["data"])
    if recomputed_audit["passed_frozen_contract"] is not True:
        raise RuntimeError(
            "Current processed data fail a fresh count audit: "
            f"{recomputed_audit['deviations']}"
        )

    recomputed_scale = compute_s0(
        run_a["r_cert"].to_numpy(dtype=float)
    )
    if not np.isclose(
        recomputed_scale,
        float(config["data"]["transforms"]["amount_scale"]),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Current processed data fail the frozen amount-scale assertion."
        )

    lower_q, upper_q = config["data"]["rt_sensitivity"][
        "quantiles_from_run_a"
    ]
    run_a_rt = run_a.loc[
        run_a["rt_included"], "rt_seconds"
    ].to_numpy(dtype=float)
    current_cutoffs = np.quantile(run_a_rt, [lower_q, upper_q])

    if not np.allclose(
        current_cutoffs,
        np.asarray(resolved_rt_cutoffs, dtype=float),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Current processed data do not reproduce the frozen RT cutoffs."
        )

    run_a_fits = pd.read_csv(
        run_a_fits_path,
        dtype={"participant": "string"},
    )

    required_metadata = {
        "fit_run",
        "multistarts_used",
        "config_sha256",
        "processed_data_sha256",
        "data_pipeline_sha256",
        "git_commit",
        "runtime_sha256",
    }
    missing_metadata = sorted(
        required_metadata - set(run_a_fits.columns)
    )
    if missing_metadata:
        raise RuntimeError(
            f"Run-A fit artifact lacks provenance columns: {missing_metadata}"
        )

    if set(run_a_fits["fit_run"].astype(str)) != {"A"}:
        raise RuntimeError(
            "Run-A fit artifact contains rows not labelled fit_run=A."
        )

    if set(run_a_fits["multistarts_used"].astype(int)) != {
        int(config["optimization"]["multistarts"])
    }:
        raise RuntimeError(
            "Run-A fit artifact used a non-frozen multistart count."
        )

    if set(run_a_fits["config_sha256"].astype(str)) != {
        run_a_config_hash
    }:
        raise RuntimeError(
            "Run-A fits do not match the configuration recorded in "
            "the Run-A completion receipt."
        )

    if set(run_a_fits["processed_data_sha256"].astype(str)) != {
        processed_hash
    }:
        raise RuntimeError(
            "Run-A fits were generated from a different processed dataset."
        )

    if set(run_a_fits["data_pipeline_sha256"].astype(str)) != {
        data_pipeline_hash
    }:
        raise RuntimeError(
            "Run-A fits were generated by a different data pipeline."
        )

    if set(run_a_fits["git_commit"].astype(str)) != {
        run_a_commit
    }:
        raise RuntimeError(
            "Run-A fits do not match the code commit recorded in "
            "the Run-A completion receipt."
        )

    if set(run_a_fits["runtime_sha256"].astype(str)) != {
        fingerprint["runtime_sha256"]
    }:
        raise RuntimeError(
            "Run-A fits were generated in a different numerical runtime."
        )

    participants = sorted(
        trials["participant"].astype(str).unique()
    )
    expected_participants = int(
        config["data"]["expected"]["participants"]
    )
    if len(participants) != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} participants, "
            f"found {len(participants)}."
        )

    try:
        full_fits, baseline_fits = validate_run_a_fit_artifact(
            run_a_fits,
            participants,
            config,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid Run-A fit artifact: {exc}"
        ) from exc

    if sha256_file(processed_path) != processed_hash:
        raise RuntimeError(
            "Processed data changed while formal preflight loaded it."
        )

    if (
        sha256_file(run_a_fits_path)
        != run_a_artifact_hashes["run_a_fits.csv"]
    ):
        raise RuntimeError(
            "Run-A fits changed while formal preflight loaded them."
        )

    final_preflight = {
        "git_commit": (
            clean_git_commit(root),
            commit,
        ),
        "config_sha256": (
            sha256_file(config["_config_path"]),
            config_hash,
        ),
        "audit_sha256": (
            sha256_file(audit_path),
            audit_hash,
        ),
        "run_a_receipt_sha256": (
            sha256_file(run_a_receipt_path),
            run_a_receipt_hash,
        ),
        "data_pipeline_sha256": (
            data_pipeline_source_hash(root),
            data_pipeline_hash,
        ),
    }
    changed_preflight = {
        name: pair
        for name, pair in final_preflight.items()
        if pair[0] != pair[1]
    }
    if changed_preflight:
        raise RuntimeError(
            f"Formal inputs changed during preflight: {changed_preflight}"
        )

    output_parent = repository_path(config, "results")
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_parent / "formal_run_b"

    if output_dir.exists():
        raise RuntimeError(
            "Formal run-B output directory already exists; refusing overwrite."
        )

    lock_directory.mkdir()
    reservation = {
        "status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
    }
    atomic_write_json(
        lock_directory / "reservation.json",
        reservation,
    )
    atomic_write_json(status_path, reservation)

    score_rows = []

    for _, fit_row in full_fits.iterrows():
        participant = str(fit_row["participant"])
        model = str(fit_row["model"]).upper()

        participant_a = run_a.loc[
            run_a["participant"].astype(str) == participant
        ]
        participant_b = run_b.loc[
            run_b["participant"].astype(str) == participant
        ]

        estimates = parameter_dict(
            fit_row,
            FULL_PARAMETER_NAMES,
        )

        prediction_a = predict_from_estimates(
            participant_a,
            estimates,
            model,
            config,
        )
        prediction_b = predict_from_estimates(
            participant_b,
            estimates,
            model,
            config,
        )

        scores = trial_scores(
            participant_b["choice_uncertain"].to_numpy(),
            participant_b["rt_seconds"].to_numpy(),
            prediction_b["choice_logit"].to_numpy(),
            prediction_b["rt_mu"].to_numpy(),
            float(prediction_b["rt_sigma"].iloc[0]),
            choice_included=participant_b[
                "choice_included"
            ].to_numpy(dtype=bool),
            rt_included=participant_b[
                "rt_included"
            ].to_numpy(dtype=bool),
        )

        scores.insert(
            0,
            "trial_index",
            participant_b["trial_index"].to_numpy(),
        )
        scores.insert(
            0,
            "condition",
            participant_b["condition"].to_numpy(),
        )
        scores.insert(
            0,
            "participant",
            participant,
        )
        scores.insert(
            0,
            "model",
            model,
        )

        scores["out_of_support"] = support_shift_flags(
            prediction_a["rt_predictor"].to_numpy(),
            prediction_b["rt_predictor"].to_numpy(),
        )

        score_rows.append(scores)

    for _, fit_row in baseline_fits.iterrows():
        participant = str(fit_row["participant"])

        participant_b = run_b.loc[
            run_b["participant"].astype(str) == participant
        ]

        prediction = predict_from_estimates(
            participant_b,
            parameter_dict(
                fit_row,
                CHOICE_PARAMETER_NAMES,
            ),
            "M1",
            config,
            choice_only=True,
        )

        include = participant_b[
            "choice_included"
        ].to_numpy(dtype=bool)

        choice_score = np.full(
            len(participant_b),
            np.nan,
        )
        brier_score = np.full(
            len(participant_b),
            np.nan,
        )
        choice_correct = np.full(
            len(participant_b),
            np.nan,
        )

        probability = choice_probability(
            prediction["choice_logit"].to_numpy()
        )

        choice_score[include] = bernoulli_logpmf_from_logits(
            participant_b.loc[
                include,
                "choice_uncertain",
            ].to_numpy(),
            prediction.loc[
                include,
                "choice_logit",
            ].to_numpy(),
        )

        observed_choice = participant_b.loc[
            include,
            "choice_uncertain",
        ].to_numpy()

        brier_score[include] = (
            probability[include] - observed_choice
        ) ** 2

        choice_correct[include] = (
            (probability[include] >= 0.5)
            == observed_choice
        ).astype(float)

        baseline_scores = pd.DataFrame(
            {
                "model": "choice_only",
                "participant": participant,
                "condition": participant_b[
                    "condition"
                ].to_numpy(),
                "trial_index": participant_b[
                    "trial_index"
                ].to_numpy(),
                "choice_log_score": choice_score,
                "brier_score": brier_score,
                "choice_correct": choice_correct,
            }
        )

        score_rows.append(baseline_scores)

    score_frame = pd.concat(
        score_rows,
        ignore_index=True,
        sort=False,
    )

    validate_score_artifact(
        score_frame,
        run_b,
        participants,
    )

    with tempfile.TemporaryDirectory(
        prefix=".formal_run_b_staging_",
        dir=output_parent,
    ) as staging_name:
        staging = Path(staging_name)

        score_frame.to_csv(
            staging / "trial_scores.csv",
            index=False,
        )

        reliability_results = []

        reliability_results.extend(
            fit_dataset(
                run_b,
                "M1",
                config,
                choice_only=True,
            )
        )

        for model in ("M1", "M2", "M3"):
            reliability_results.extend(
                fit_dataset(
                    run_b,
                    model,
                    config,
                )
            )

        (
            reliability_frame,
            reliability_failure_count,
        ) = validate_reliability_matrix(
            reliability_results,
            participants,
            config,
        )

        failed_reliability = [
            (
                result.participant,
                result.model,
            )
            for result in reliability_results
            if not result.success
        ]

        reliability_frame.to_csv(
            staging / "run_b_reliability_fits.csv",
            index=False,
        )

        start_diagnostics_frame(
            reliability_results
        ).to_csv(
            staging / "run_b_optimizer_starts.csv",
            index=False,
        )

        staging.rename(output_dir)

    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.glob("*.csv"))
    }

    completed = {
        "status": "completed",
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "fingerprint": fingerprint,
        "output_sha256": output_hashes,
        "run_b_reliability_failures": [
            list(item)
            for item in failed_reliability
        ],
        "run_b_reliability_failure_count": (
            reliability_failure_count
        ),
    }

    append_manifest(
        repository_path(
            config,
            config["outputs"]["manifest"],
        ),
        [
            {
                "artifact": "formal_run_b_trial_scores",
                "stage": "run_b_once",
                "timestamp_utc": completed[
                    "completed_at_utc"
                ],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "raw_archive_sha256": (
                    current_archive_hash or ""
                ),
                "raw_source_mode": audit[
                    "raw_source_mode"
                ],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": data_pipeline_hash,
                "artifact_sha256": output_hashes[
                    "trial_scores.csv"
                ],
                "run": "B",
                "fit_status": "passed",
                "path": (
                    "results/formal_run_b/"
                    "trial_scores.csv"
                ),
            },
            {
                "artifact": "run_b_reliability_fits",
                "stage": "run_b_once",
                "timestamp_utc": completed[
                    "completed_at_utc"
                ],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "raw_archive_sha256": (
                    current_archive_hash or ""
                ),
                "raw_source_mode": audit[
                    "raw_source_mode"
                ],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": data_pipeline_hash,
                "artifact_sha256": output_hashes[
                    "run_b_reliability_fits.csv"
                ],
                "run": "B",
                "fit_status": (
                    "passed"
                    if not failed_reliability
                    else "failures_present"
                ),
                "path": (
                    "results/formal_run_b/"
                    "run_b_reliability_fits.csv"
                ),
            },
            {
                "artifact": "run_b_optimizer_starts",
                "stage": "run_b_once",
                "timestamp_utc": completed[
                    "completed_at_utc"
                ],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "raw_archive_sha256": (
                    current_archive_hash or ""
                ),
                "raw_source_mode": audit[
                    "raw_source_mode"
                ],
                "processed_data_sha256": processed_hash,
                "data_pipeline_sha256": data_pipeline_hash,
                "artifact_sha256": output_hashes[
                    "run_b_optimizer_starts.csv"
                ],
                "run": "B",
                "fit_status": (
                    "passed"
                    if not failed_reliability
                    else "failures_present"
                ),
                "path": (
                    "results/formal_run_b/"
                    "run_b_optimizer_starts.csv"
                ),
            },
        ],
    )

    atomic_write_json(
        lock_directory / "reservation.json",
        completed,
    )
    atomic_write_json(
        status_path,
        completed,
    )

    print(
        f"Formal run-B outputs written once under {output_dir}"
    )


if __name__ == "__main__":
    main()