#!/usr/bin/env python3
"""One-shot held-out scoring and independently initialized run-B reliability fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
)
from pd_project.likelihood import bernoulli_logpmf_from_logits
from pd_project.provenance import (
    append_manifest,
    clean_git_commit,
    data_transform_contract_hash,
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
    raw_directory = repository_path(config, config["data"]["raw_directory"])
    raw_files = sorted(raw_directory.glob(config["data"].get("raw_glob", "*.mat")))
    raw_hash = sha256_named_files(raw_files)
    processed_path = repository_path(config, config["data"]["processed_trials"])
    run_a_fits_path = repository_path(config, "results/run_a_fits.csv")
    audit_path = repository_path(config, config["data"]["audit_report"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("passed_expected_counts", False)):
        raise RuntimeError("Formal run B requires a data audit with all expected counts passed.")
    if not bool(audit.get("amount_scale_matches_expected", False)):
        raise RuntimeError("Formal run B requires the run-A amount-scale assertion to pass.")
    resolved_rt_cutoffs = config["data"]["rt_sensitivity"].get("resolved_seconds")
    if not isinstance(resolved_rt_cutoffs, list) or len(resolved_rt_cutoffs) != 2:
        raise RuntimeError("Formal run B requires two frozen resolved RT sensitivity cutoffs.")
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
        raise RuntimeError("Frozen RT sensitivity cutoffs do not match the run-A audit.")
    processed_hash = sha256_file(processed_path)
    data_config_hash = data_transform_contract_hash(config["data"])
    provenance_mismatches = {
        "raw_data_sha256": (audit.get("raw_data_sha256"), raw_hash),
        "processed_data_sha256": (
            audit.get("processed_data_sha256"),
            processed_hash,
        ),
        "data_config_sha256": (audit.get("data_config_sha256"), data_config_hash),
    }
    provenance_mismatches = {
        name: values for name, values in provenance_mismatches.items() if values[0] != values[1]
    }
    if provenance_mismatches:
        raise RuntimeError(
            f"Data audit provenance does not match current inputs: {provenance_mismatches}"
        )
    fingerprint = {
        "config_sha256": config_hash,
        "git_commit": commit,
        "raw_data_sha256": raw_hash,
        "processed_data_sha256": processed_hash,
        "run_a_fits_sha256": sha256_file(run_a_fits_path),
        "runtime_sha256": sha256_mapping(runtime_metadata()),
    }

    # The local directory creation is atomic, preventing concurrent processes
    # in this checkout. The tracked registry prevents a committed completed run
    # from silently becoming runnable again in a fresh clone.
    lock_directory.mkdir()
    reservation = {
        "status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
    }
    (lock_directory / "reservation.json").write_text(
        json.dumps(reservation, indent=2, sort_keys=True), encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trials = pd.read_csv(
        processed_path,
        dtype={"participant": "string"},
    )
    run_a = trials.loc[trials["run"] == "A"].copy()
    run_b = trials.loc[trials["run"] == "B"].copy()
    recomputed_audit = audit_trials(trials, config["data"])
    if not recomputed_audit["passed_expected_counts"]:
        raise RuntimeError(
            f"Current processed data fail a fresh count audit: {recomputed_audit['deviations']}"
        )
    recomputed_scale = compute_s0(run_a["r_cert"].to_numpy(dtype=float))
    if not np.isclose(
        recomputed_scale,
        float(config["data"]["transforms"]["amount_scale"]),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Current processed data fail the frozen amount-scale assertion.")
    lower_q, upper_q = config["data"]["rt_sensitivity"]["quantiles_from_run_a"]
    run_a_rt = run_a.loc[run_a["rt_included"], "rt_seconds"].to_numpy(dtype=float)
    current_cutoffs = np.quantile(run_a_rt, [lower_q, upper_q])
    if not np.allclose(
        current_cutoffs,
        np.asarray(resolved_rt_cutoffs, dtype=float),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Current processed data do not reproduce the frozen RT cutoffs.")
    run_a_fits = pd.read_csv(
        run_a_fits_path,
        dtype={"participant": "string"},
    )
    choice_only_flag = run_a_fits["choice_only"].astype(str).str.lower().eq("true")
    success_flag = run_a_fits["success"].astype(str).str.lower().eq("true")
    if not bool(success_flag.all()):
        failed = run_a_fits.loc[~success_flag, ["participant", "model"]]
        raise RuntimeError(
            "Formal run B is blocked by unsuccessful run-A fits: "
            + failed.to_dict(orient="records").__repr__()
        )
    required_metadata = {
        "fit_run",
        "multistarts_used",
        "config_sha256",
        "processed_data_sha256",
        "git_commit",
        "runtime_sha256",
    }
    missing_metadata = sorted(required_metadata - set(run_a_fits.columns))
    if missing_metadata:
        raise RuntimeError(f"Run-A fit artifact lacks provenance columns: {missing_metadata}")
    if set(run_a_fits["fit_run"].astype(str)) != {"A"}:
        raise RuntimeError("Run-A fit artifact contains rows not labelled fit_run=A.")
    if set(run_a_fits["multistarts_used"].astype(int)) != {
        int(config["optimization"]["multistarts"])
    }:
        raise RuntimeError("Run-A fit artifact used a non-frozen multistart count.")
    if set(run_a_fits["config_sha256"].astype(str)) != {config_hash}:
        raise RuntimeError("Run-A fits were not generated by the current frozen config.")
    if set(run_a_fits["processed_data_sha256"].astype(str)) != {processed_hash}:
        raise RuntimeError("Run-A fits were generated from a different processed dataset.")
    if set(run_a_fits["git_commit"].astype(str)) != {commit}:
        raise RuntimeError("Run-A fits were generated from a different code commit.")
    if set(run_a_fits["runtime_sha256"].astype(str)) != {
        fingerprint["runtime_sha256"]
    }:
        raise RuntimeError("Run-A fits were generated in a different numerical runtime.")
    full_fits = run_a_fits.loc[~choice_only_flag].copy()
    baseline_fits = run_a_fits.loc[choice_only_flag].copy()
    participants = sorted(trials["participant"].astype(str).unique())
    expected_participants = int(config["data"]["expected"]["participants"])
    if len(participants) != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} participants, found {len(participants)}."
        )
    expected_pairs = {
        (participant, model)
        for participant in participants
        for model in ("choice_only", "M1", "M2", "M3")
    }
    observed_pair_counts = (
        run_a_fits.assign(
            participant=run_a_fits["participant"].astype(str),
            model=run_a_fits["model"].astype(str),
        )
        .groupby(["participant", "model"])
        .size()
    )
    observed_pairs = set(observed_pair_counts.index)
    duplicates = observed_pair_counts.loc[observed_pair_counts != 1]
    if observed_pairs != expected_pairs or not duplicates.empty:
        raise RuntimeError(
            "Run-A fit matrix must contain exactly one choice_only/M1/M2/M3 row "
            f"per participant. Missing={sorted(expected_pairs - observed_pairs)}, "
            f"extra={sorted(observed_pairs - expected_pairs)}, "
            f"nonunique={duplicates.to_dict()}"
        )

    score_rows = []
    for _, fit_row in full_fits.iterrows():
        participant = str(fit_row["participant"])
        model = str(fit_row["model"]).upper()
        participant_a = run_a.loc[run_a["participant"].astype(str) == participant]
        participant_b = run_b.loc[run_b["participant"].astype(str) == participant]
        estimates = parameter_dict(fit_row, FULL_PARAMETER_NAMES)
        prediction_a = predict_from_estimates(participant_a, estimates, model, config)
        prediction_b = predict_from_estimates(participant_b, estimates, model, config)
        scores = trial_scores(
            participant_b["choice_uncertain"].to_numpy(),
            participant_b["rt_seconds"].to_numpy(),
            prediction_b["choice_logit"].to_numpy(),
            prediction_b["rt_mu"].to_numpy(),
            float(prediction_b["rt_sigma"].iloc[0]),
            choice_included=participant_b["choice_included"].to_numpy(dtype=bool),
            rt_included=participant_b["rt_included"].to_numpy(dtype=bool),
        )
        scores.insert(0, "trial_index", participant_b["trial_index"].to_numpy())
        scores.insert(0, "condition", participant_b["condition"].to_numpy())
        scores.insert(0, "participant", participant)
        scores.insert(0, "model", model)
        scores["out_of_support"] = support_shift_flags(
            prediction_a["rt_predictor"].to_numpy(),
            prediction_b["rt_predictor"].to_numpy(),
        )
        score_rows.append(scores)

    for _, fit_row in baseline_fits.iterrows():
        participant = str(fit_row["participant"])
        participant_b = run_b.loc[run_b["participant"].astype(str) == participant]
        prediction = predict_from_estimates(
            participant_b,
            parameter_dict(fit_row, CHOICE_PARAMETER_NAMES),
            "M1",
            config,
            choice_only=True,
        )
        include = participant_b["choice_included"].to_numpy(dtype=bool)
        choice_score = np.full(len(participant_b), np.nan)
        brier_score = np.full(len(participant_b), np.nan)
        choice_correct = np.full(len(participant_b), np.nan)
        probability = choice_probability(prediction["choice_logit"].to_numpy())
        choice_score[include] = bernoulli_logpmf_from_logits(
            participant_b.loc[include, "choice_uncertain"].to_numpy(),
            prediction.loc[include, "choice_logit"].to_numpy(),
        )
        observed_choice = participant_b.loc[include, "choice_uncertain"].to_numpy()
        brier_score[include] = (probability[include] - observed_choice) ** 2
        choice_correct[include] = (
            (probability[include] >= 0.5) == observed_choice
        ).astype(float)
        baseline_scores = pd.DataFrame(
            {
                "model": "choice_only",
                "participant": participant,
                "condition": participant_b["condition"].to_numpy(),
                "trial_index": participant_b["trial_index"].to_numpy(),
                "choice_log_score": choice_score,
                "brier_score": brier_score,
                "choice_correct": choice_correct,
            }
        )
        score_rows.append(baseline_scores)

    output_parent = repository_path(config, "results")
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_parent / "formal_run_b"
    if output_dir.exists():
        raise RuntimeError("Formal run-B output directory already exists; refusing overwrite.")
    with tempfile.TemporaryDirectory(
        prefix=".formal_run_b_staging_", dir=output_parent
    ) as staging_name:
        staging = Path(staging_name)
        pd.concat(score_rows, ignore_index=True, sort=False).to_csv(
            staging / "trial_scores.csv", index=False
        )

        reliability_results = []
        reliability_results.extend(fit_dataset(run_b, "M1", config, choice_only=True))
        for model in ("M1", "M2", "M3"):
            reliability_results.extend(fit_dataset(run_b, model, config))
        failed_reliability = [
            (result.participant, result.model)
            for result in reliability_results
            if not result.success
        ]
        fit_results_frame(reliability_results).to_csv(
            staging / "run_b_reliability_fits.csv", index=False
        )
        start_diagnostics_frame(reliability_results).to_csv(
            staging / "run_b_optimizer_starts.csv", index=False
        )
        staging.rename(output_dir)

    output_hashes = {
        path.name: sha256_file(path) for path in sorted(output_dir.glob("*.csv"))
    }
    completed = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "output_sha256": output_hashes,
        "run_b_reliability_failures": [list(item) for item in failed_reliability],
    }
    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        [
            {
                "artifact": "formal_run_b_trial_scores",
                "stage": "run_b_once",
                "timestamp_utc": completed["completed_at_utc"],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "processed_data_sha256": processed_hash,
                "artifact_sha256": output_hashes["trial_scores.csv"],
                "run": "B",
                "fit_status": "passed",
                "path": "results/formal_run_b/trial_scores.csv",
            },
            {
                "artifact": "run_b_reliability_fits",
                "stage": "run_b_once",
                "timestamp_utc": completed["completed_at_utc"],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "processed_data_sha256": processed_hash,
                "artifact_sha256": output_hashes["run_b_reliability_fits.csv"],
                "run": "B",
                "fit_status": (
                    "passed" if not failed_reliability else "failures_present"
                ),
                "path": "results/formal_run_b/run_b_reliability_fits.csv",
            },
            {
                "artifact": "run_b_optimizer_starts",
                "stage": "run_b_once",
                "timestamp_utc": completed["completed_at_utc"],
                "git_commit": commit,
                "config_sha256": config_hash,
                "raw_data_sha256": raw_hash,
                "processed_data_sha256": processed_hash,
                "artifact_sha256": output_hashes["run_b_optimizer_starts.csv"],
                "run": "B",
                "fit_status": (
                    "passed" if not failed_reliability else "failures_present"
                ),
                "path": "results/formal_run_b/run_b_optimizer_starts.csv",
            },
        ],
    )
    # Completion is written last, after output hashing and manifest creation.
    (lock_directory / "reservation.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Formal run-B outputs written once under {output_dir}")


if __name__ == "__main__":
    main()
