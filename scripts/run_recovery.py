#!/usr/bin/env python3
"""Run synthetic smoke checks or the configured formal recovery experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from pd_project.config import load_config, repository_path, sha256_file
from pd_project.data import audit_trials, compute_s0, odds_against
from pd_project.fit import (
    CHOICE_PARAMETER_NAMES,
    FULL_PARAMETER_NAMES,
    FitResult,
    deterministic_seed,
    fit_participant,
    predict_from_estimates,
)
from pd_project.likelihood import lognormal_logpdf
from pd_project.provenance import (
    append_manifest,
    clean_git_commit,
    runtime_metadata,
    sha256_mapping,
)
from pd_project.recovery import (
    model_recovery_confusion,
    recovery_metrics,
    simulate_participant,
)
from pd_project.rt_models import rt_predictor
from pd_project.valuation_choice import parameter_by_condition, subjective_values


MODELS = ("M1", "M2", "M3")
DESIGN_COLUMNS = (
    "participant", "run", "trial_index", "condition",
    "r_cert", "r_uncert", "probability", "odds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--confirm-design-frozen",
        action="store_true",
        help="Required with --formal after the group approves recovery ranges and counts.",
    )
    return parser.parse_args()


def synthetic_design(run: str) -> pd.DataFrame:
    """Create a small balanced design used only by the CI smoke path."""

    if run not in {"A", "B"}:
        raise ValueError("Synthetic smoke run must be A or B.")
    probability = np.tile(np.array([0.10, 0.25, 0.50, 0.75, 0.90]), 8)
    n_trials = probability.size
    condition = np.where(np.arange(n_trials) % 2 == 0, "R", "L")
    signs = np.where(condition == "R", 1.0, -1.0)
    return pd.DataFrame(
        {
            "participant": "smoke_001",
            "run": run,
            "trial_index": np.arange(1, n_trials + 1),
            "condition": condition,
            "r_cert": signs * 10.0,
            "r_uncert": signs * np.linspace(12.0, 30.0, n_trials),
            "probability": probability,
            "odds": odds_against(probability),
        }
    )


def best_start_near_boundary(fit: FitResult) -> bool | None:
    if fit.best_start_index < 0:
        return None
    return bool(fit.starts[fit.best_start_index].near_boundary)


def predictor_from_estimates(
    trials: pd.DataFrame,
    estimates: dict[str, float],
    model: str,
    config: dict[str, Any],
) -> np.ndarray:
    k = parameter_by_condition(
        trials["condition"].to_numpy(),
        np.exp(estimates["log_k_R"]),
        np.exp(estimates["log_k_L"]),
    )
    values = subjective_values(
        trials["r_cert"].to_numpy(dtype=float),
        trials["r_uncert"].to_numpy(dtype=float),
        trials["odds"].to_numpy(dtype=float),
        k,
        s0=float(config["data"]["transforms"]["amount_scale"]),
    )
    return rt_predictor(
        model,
        values.v_cert,
        values.v_uncert,
        values.delta_v,
    )


def score_rt_by_condition(
    simulated_a: pd.DataFrame,
    simulated_b: pd.DataFrame,
    fit: FitResult,
    model: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score one A-fit model on the matching synthetic B trials."""

    if not fit.success:
        return [
            {
                "condition": condition,
                "run_b_rt_mlpd": np.nan,
                "n_rt": int((simulated_b["condition"] == condition).sum()),
                "out_of_support_trial_fraction": np.nan,
                "n_in_support": np.nan,
                "n_out_of_support": np.nan,
                "in_support_rt_mlpd": np.nan,
                "out_of_support_rt_mlpd": np.nan,
            }
            for condition in ("R", "L")
        ]

    prediction = predict_from_estimates(simulated_b, fit.estimates, model, config)
    include_rt = simulated_b["rt_included"].to_numpy(copy=False)
    if include_rt.dtype != np.dtype(bool):
        raise ValueError("Synthetic RT mask must have boolean dtype.")

    scores = lognormal_logpdf(
        simulated_b.loc[include_rt, "rt_seconds"].to_numpy(dtype=float),
        prediction.loc[include_rt, "rt_mu"].to_numpy(dtype=float),
        float(prediction["rt_sigma"].iloc[0]),
    )

    predictor_a = predictor_from_estimates(
        simulated_a,
        fit.estimates,
        model,
        config,
    )
    predictor_b = predictor_from_estimates(
        simulated_b,
        fit.estimates,
        model,
        config,
    )

    predictor_a_min = float(np.min(predictor_a))
    predictor_a_max = float(np.max(predictor_a))
    scored_predictor_b = predictor_b[include_rt]

    out_of_support = (
        (scored_predictor_b < predictor_a_min)
        | (scored_predictor_b > predictor_a_max)
    )

    scored = pd.DataFrame(
        {
            "condition": simulated_b.loc[
                include_rt, "condition"
            ].to_numpy(dtype=str),
            "rt_log_score": scores,
            "out_of_support": out_of_support,
        }
    )

    rows = []
    for condition in ("R", "L"):
        condition_scores = scored.loc[
            scored["condition"] == condition
        ].copy()

        if condition_scores.empty:
            raise ValueError(
                f"Synthetic B has no scored RT trials for condition {condition}."
            )

        in_support = condition_scores.loc[
            ~condition_scores["out_of_support"],
            "rt_log_score",
        ]
        out_support = condition_scores.loc[
            condition_scores["out_of_support"],
            "rt_log_score",
        ]

        rows.append(
            {
                "condition": condition,
                "run_b_rt_mlpd": float(
                    condition_scores["rt_log_score"].mean()
                ),
                "n_rt": int(len(condition_scores)),
                "out_of_support_trial_fraction": float(
                    condition_scores["out_of_support"].mean()
                ),
                "n_in_support": int(len(in_support)),
                "n_out_of_support": int(len(out_support)),
                "in_support_rt_mlpd": float(in_support.mean())
                if not in_support.empty else np.nan,
                "out_of_support_rt_mlpd": float(out_support.mean())
                if not out_support.empty else np.nan,
            }
        )

    return rows


def run_smoke(config: dict[str, Any]) -> None:
    """Exercise A simulation, A fitting, and held-out B scoring for all models."""

    master_seed = int(config["random"]["master_seed"])
    design_a = synthetic_design("A")
    design_b = synthetic_design("B")
    parameters = {
        "log_k_R": np.log(1.0), "log_k_L": np.log(2.0),
        "log_beta": np.log(3.0), "alpha": np.log(2.0),
        "delta_L": 0.15, "log_b": np.log(0.4),
        "log_sigma": np.log(0.35),
    }
    rows = []
    for generating_model in MODELS:
        simulated_a = simulate_participant(
            design_a, parameters, generating_model,
            rng=np.random.default_rng(
                deterministic_seed(master_seed, "recovery_smoke", generating_model, "A")
            ),
            s0=10.0,
        )
        simulated_b = simulate_participant(
            design_b, parameters, generating_model,
            rng=np.random.default_rng(
                deterministic_seed(master_seed, "recovery_smoke", generating_model, "B")
            ),
            s0=10.0,
        )
        baseline = fit_participant(
            simulated_a, "M1", config,
            seed=deterministic_seed(
                master_seed, "recovery_smoke", generating_model, "choice_only"
            ),
            choice_only=True, multistarts=2,
        )
        if not baseline.success:
            raise RuntimeError(
                f"{generating_model} choice-only smoke fit failed: {baseline.message}"
            )
        for fitted_model in MODELS:
            fit = fit_participant(
                simulated_a, fitted_model, config,
                seed=deterministic_seed(
                    master_seed, "recovery_smoke", generating_model, fitted_model
                ),
                multistarts=2,
            )
            if not fit.success:
                raise RuntimeError(
                    f"{generating_model}->{fitted_model} smoke fit failed: "
                    f"objective={fit.objective}, message={fit.message}"
                )
            for score in score_rt_by_condition(
                simulated_a,
                simulated_b,
                fit,
                fitted_model,
                config,
            ):
                rows.append(
                    {
                        "generating_model": generating_model,
                        "replicate": 0,
                        "fitted_model": fitted_model,
                        **score,
                    }
                )
    confusion = model_recovery_confusion(pd.DataFrame(rows), expected_replicates=1)
    if confusion.shape != (3, 3) or not np.allclose(confusion.sum(axis=1), 1.0):
        raise RuntimeError("Smoke model-recovery confusion matrix is invalid.")
    print("Smoke A-fit/B-score recovery succeeded for M1, M2, and M3.")
    print(confusion.to_string())


def load_formal_design(config: dict[str, Any]) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Load a freshly audited canonical design without consuming observed outcomes."""

    processed_path = repository_path(config, config["data"]["processed_trials"])
    audit_path = repository_path(config, config["data"]["audit_report"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_passes = (
        "passed_integrity_checks", "passed_frozen_contract",
        "approved_for_fitting", "amount_scale_matches_expected",
    )
    failed = [name for name in required_passes if audit.get(name) is not True]
    if failed:
        raise RuntimeError(f"Formal recovery requires a canonical data audit: {failed}")
    processed_hash = sha256_file(processed_path)
    if audit.get("processed_data_sha256") != processed_hash:
        raise RuntimeError("Processed data do not match the canonical audit.")
    trials = pd.read_csv(processed_path, dtype={"participant": "string"})
    if sha256_file(processed_path) != processed_hash:
        raise RuntimeError("Processed data changed while recovery loaded them.")
    fresh = audit_trials(trials, config["data"])
    if fresh["passed_frozen_contract"] is not True:
        raise RuntimeError(f"Processed data fail a fresh audit: {fresh['deviations']}")
    scale = compute_s0(
        trials.loc[trials["run"] == "A", "r_cert"].to_numpy(dtype=float)
    )
    if not np.isclose(
        scale, float(config["data"]["transforms"]["amount_scale"]),
        atol=1.0e-12, rtol=0.0,
    ):
        raise RuntimeError("Formal recovery design fails the frozen amount scale.")
    missing_design = sorted(set(DESIGN_COLUMNS) - set(trials.columns))
    if missing_design:
        raise RuntimeError(f"Processed data lack recovery design columns: {missing_design}")
    return trials.loc[:, DESIGN_COLUMNS].copy(), scale, audit


def load_run_a_truths(
    config: dict[str, Any],
    participants: list[str],
    audit: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    root = Path(config["_repository_root"])
    path = root / "results" / "run_a_fits.csv"
    if not path.is_file():
        raise RuntimeError("Formal recovery requires results/run_a_fits.csv.")

    fits = pd.read_csv(path, dtype={"participant": "string"})
    full = fits.loc[
        (~fits["choice_only"].astype(bool))
        & fits["model"].isin(MODELS)
    ].copy()

    if not bool(full["success"].astype(bool).all()):
        raise RuntimeError("Formal recovery requires successful Run-A MAP fits.")

    if not bool((full["fit_run"].astype(str) == "A").all()):
        raise RuntimeError("Formal recovery truth source must contain Run-A fits only.")

    expected = {(participant, model) for participant in participants for model in MODELS}
    observed = {
        (str(row.participant), str(row.model))
        for row in full.itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError("Run-A fits do not contain exactly one full fit per participant/model.")

    if full.duplicated(["participant", "model"]).any():
        raise RuntimeError("Run-A fits contain duplicate participant/model rows.")

    if set(full["processed_data_sha256"].astype(str)) != {
        str(audit["processed_data_sha256"])
    }:
        raise RuntimeError("Run-A fits do not match the canonical processed data.")

    truths: dict[tuple[str, str], dict[str, float]] = {}
    for row in full.itertuples(index=False):
        parameters = {
            name: float(getattr(row, name))
            for name in FULL_PARAMETER_NAMES
        }
        if not all(np.isfinite(value) for value in parameters.values()):
            raise RuntimeError(
                f"Run-A truth contains non-finite parameters for "
                f"{row.participant}/{row.model}."
            )
        truths[(str(row.participant), str(row.model))] = parameters

    return truths


def fit_record(
    fit: FitResult,
    *,
    generating_model: str,
    fitted_model: str,
    replicate: int,
    sample_kind: str,
    source_participant: str,
    true_parameters: dict[str, float],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "generating_model": generating_model,
        "fitted_model": fitted_model,
        "replicate": replicate,
        "sample_kind": sample_kind,
        "source_participant": source_participant,
        "success": fit.success,
        "objective": fit.objective,
        "best_start_index": fit.best_start_index,
        "near_boundary": best_start_near_boundary(fit),
        "message": fit.message,
    }
    record.update({f"true_{name}": value for name, value in true_parameters.items()})
    record.update({f"estimate_{name}": value for name, value in fit.estimates.items()})
    return record


def summarize_parameter_recovery(
    fits: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize nominal generating-model and choice-only parameter recovery."""

    rows = []
    nominal = fits.loc[fits["sample_kind"] == "nominal"].copy()
    for generating_model in MODELS:
        specifications = [
            (generating_model, FULL_PARAMETER_NAMES, "generating_full"),
            ("choice_only", CHOICE_PARAMETER_NAMES, "choice_only"),
        ]
        for fitted_model, parameters, fit_type in specifications:
            subset = nominal.loc[
                (nominal["generating_model"] == generating_model)
                & (nominal["fitted_model"] == fitted_model)
            ].sort_values("replicate")
            success = subset["success"].to_numpy(dtype=bool)
            boundary = subset["near_boundary"].fillna(False).to_numpy(dtype=bool)
            for parameter in parameters:
                true_values = subset[f"true_{parameter}"].to_numpy(dtype=float)
                parameter_range = float(np.max(true_values) - np.min(true_values))
                if not np.isfinite(parameter_range) or parameter_range <= 0.0:
                    raise ValueError(
                        f"Nominal truth range for {generating_model}/{parameter} "
                        "must be finite and positive."
                    )
                metrics = recovery_metrics(
                    true_values,
                    subset[f"estimate_{parameter}"].to_numpy(dtype=float),
                    parameter_range=parameter_range,
                    success_mask=success,
                )
                rows.append(
                    {
                        "generating_model": generating_model,
                        "fit_type": fit_type,
                        "fitted_model": fitted_model,
                        "parameter": parameter,
                        "boundary_rate": float(boundary[success].mean())
                        if bool(success.any()) else np.nan,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarize_support_shift(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = scores.loc[
        scores["fit_success"].astype(bool)
    ].groupby(
        [
            "generating_model",
            "fitted_model",
            "sample_kind",
            "condition",
        ],
        sort=True,
    )

    for keys, subset in grouped:
        generating_model, fitted_model, sample_kind, condition = keys

        n_rt = int(subset["n_rt"].sum())
        n_out = int(subset["n_out_of_support"].sum())

        rows.append(
            {
                "generating_model": generating_model,
                "fitted_model": fitted_model,
                "sample_kind": sample_kind,
                "condition": condition,
                "out_of_support_trial_fraction": (
                    float(n_out / n_rt) if n_rt > 0 else np.nan
                ),
                "affected_participant_fraction": float(
                    (subset["n_out_of_support"] > 0).mean()
                ),
                "in_support_rt_mlpd": float(
                    subset["in_support_rt_mlpd"].mean()
                ),
                "out_of_support_rt_mlpd": float(
                    subset["out_of_support_rt_mlpd"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_formal(config: dict[str, Any], *, confirmed: bool) -> None:
    """Execute the configured nominal and design-stress recovery experiment."""

    if not confirmed:
        raise SystemExit("--formal requires --confirm-design-frozen.")

    root = Path(config["_repository_root"])
    commit = clean_git_commit(root)
    design, scale, audit = load_formal_design(config)
    recovery_config = config["recovery"]

    if tuple(recovery_config["generating_models"]) != MODELS:
        raise RuntimeError("Formal recovery generating_models must be M1/M2/M3.")
    if recovery_config.get("parameter_sampler") != "run_a_map_truths":
        raise RuntimeError(
            "Formal recovery parameter_sampler must be run_a_map_truths."
        )

    stress_config = recovery_config["stress_set"]
    if stress_config.get("include_parameter_edges") is not False:
        raise RuntimeError("Formal recovery parameter-edge stress must be disabled.")
    if stress_config.get("source") != "largest_abs_r_uncert":
        raise RuntimeError(
            "Formal recovery stress source must be largest_abs_r_uncert."
        )

    n_nominal = int(recovery_config["synthetic_participants_per_model"])
    n_stress = int(recovery_config["stress_participants_per_model"])
    n_stress_sources = int(stress_config["source_participants"])
    master_seed = int(config["random"]["master_seed"])

    participants = sorted(design["participant"].astype(str).unique())
    if not participants:
        raise RuntimeError("Formal recovery design contains no participants.")
    if not 1 <= n_stress_sources <= len(participants):
        raise RuntimeError("Invalid recovery stress source participant count.")

    run_a_truths = load_run_a_truths(config, participants, audit)

    stress_participants = (
        design.assign(abs_r_uncert=np.abs(design["r_uncert"].to_numpy(dtype=float)))
        .groupby("participant")["abs_r_uncert"]
        .max()
        .sort_values(ascending=False)
        .head(n_stress_sources)
        .index.astype(str)
        .tolist()
    )

    output_dir = repository_path(config, config["outputs"]["recovery_directory"])
    if output_dir.exists():
        raise RuntimeError("Recovery output directory already exists; refusing overwrite.")

    fit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    true_rows: list[dict[str, Any]] = []

    for generating_model in MODELS:
        for sample_kind, n_samples, source_pool in (
            ("nominal", n_nominal, participants),
            ("stress", n_stress, stress_participants),
        ):
            for replicate in range(n_samples):
                source_participant = source_pool[replicate % len(source_pool)]
                true_parameters = dict(
                    run_a_truths[(source_participant, generating_model)]
                )

                synthetic_id = (
                    f"{generating_model}_{sample_kind}_{replicate:04d}"
                )

                participant_design = design.loc[
                    design["participant"].astype(str) == source_participant
                ].copy()
                participant_design["participant"] = synthetic_id

                designs = {
                    run: participant_design.loc[
                        participant_design["run"] == run
                    ].copy()
                    for run in ("A", "B")
                }
                if any(frame.empty for frame in designs.values()):
                    raise RuntimeError(
                        f"Source participant {source_participant} lacks A or B design."
                    )

                simulated = {}
                for run in ("A", "B"):
                    simulation_seed = deterministic_seed(
                        master_seed,
                        "formal_recovery",
                        generating_model,
                        sample_kind,
                        replicate,
                        "simulate",
                        run,
                    )
                    seed_rows.append(
                        {
                            "generating_model": generating_model,
                            "replicate": replicate,
                            "sample_kind": sample_kind,
                            "stage": "simulate",
                            "run": run,
                            "fitted_model": "",
                            "seed": simulation_seed,
                        }
                    )
                    simulated[run] = simulate_participant(
                        designs[run],
                        true_parameters,
                        generating_model,
                        rng=np.random.default_rng(simulation_seed),
                        s0=scale,
                    )

                true_rows.append(
                    {
                        "generating_model": generating_model,
                        "replicate": replicate,
                        "sample_kind": sample_kind,
                        "source_participant": source_participant,
                        "truth_source": "run_a_map",
                        **true_parameters,
                    }
                )

                fit_specs = [("choice_only", "M1", True)] + [
                    (model, model, False) for model in MODELS
                ]

                for output_model, fit_model, choice_only in fit_specs:
                    fit_seed = deterministic_seed(
                        master_seed,
                        "formal_recovery",
                        generating_model,
                        sample_kind,
                        replicate,
                        "fit_A",
                        output_model,
                    )
                    seed_rows.append(
                        {
                            "generating_model": generating_model,
                            "replicate": replicate,
                            "sample_kind": sample_kind,
                            "stage": "fit",
                            "run": "A",
                            "fitted_model": output_model,
                            "seed": fit_seed,
                        }
                    )
                    fit = fit_participant(
                        simulated["A"],
                        fit_model,
                        config,
                        seed=fit_seed,
                        choice_only=choice_only,
                    )
                    fit_rows.append(
                        fit_record(
                            fit,
                            generating_model=generating_model,
                            fitted_model=output_model,
                            replicate=replicate,
                            sample_kind=sample_kind,
                            source_participant=source_participant,
                            true_parameters=true_parameters,
                        )
                    )

                    if not choice_only:
                        for score in score_rt_by_condition(
                            simulated["A"],
                            simulated["B"],
                            fit,
                            fit_model,
                            config,
                        ):
                            score_rows.append(
                                {
                                    "generating_model": generating_model,
                                    "replicate": replicate,
                                    "sample_kind": sample_kind,
                                    "participant": synthetic_id,
                                    "run": "B",
                                    "fitted_model": output_model,
                                    "fit_success": fit.success,
                                    **score,
                                }
                            )

    fits = pd.DataFrame(fit_rows)
    scores = pd.DataFrame(score_rows)
    seeds = pd.DataFrame(seed_rows)
    truths = pd.DataFrame(true_rows)

    issues: list[str] = []
    try:
        parameter_summary = summarize_parameter_recovery(fits, config)
    except ValueError as exc:
        parameter_summary = pd.DataFrame()
        issues.append(f"parameter_recovery: {exc}")

    try:
        confusion = model_recovery_confusion(
            scores.loc[scores["sample_kind"] == "nominal"].copy(),
            expected_replicates=n_nominal,
        )
    except ValueError as exc:
        confusion = pd.DataFrame(
            np.nan,
            index=MODELS,
            columns=MODELS,
        ).rename_axis("generating_model")
        issues.append(f"model_recovery: {exc}")

    support_shift_summary = summarize_support_shift(scores)

    timestamp = datetime.now(timezone.utc).isoformat()
    runtime = runtime_metadata()

    metadata = {
        "status": "completed" if not issues else "completed_with_issues",
        "completed_at_utc": timestamp,
        "issues": issues,
        "git_commit": commit,
        "config_sha256": sha256_file(config["_config_path"]),
        "processed_data_sha256": audit["processed_data_sha256"],
        "data_pipeline_sha256": audit["data_pipeline_sha256"],
        "runtime_sha256": sha256_mapping(runtime),
        "truth_source": "run_a_map",
        "stress_source": "largest_abs_r_uncert",
        "stress_source_participants": stress_participants,
        "nominal_participants_per_model": n_nominal,
        "stress_participants_per_model": n_stress,
        "fit_failures": int((~fits["success"].astype(bool)).sum()),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".recovery_staging_",
        dir=output_dir.parent,
    ) as staging_name:
        staging = Path(staging_name)

        artifacts = {
            "true_parameters.csv": truths,
            "fit_results.csv": fits,
            "run_b_model_scores.csv": scores,
            "parameter_recovery_metrics.csv": parameter_summary,
            "model_recovery_confusion.csv": confusion.reset_index(),
            "support_shift_summary.csv": support_shift_summary,
            "replicate_seeds.csv": seeds,
        }

        for filename, frame in artifacts.items():
            frame.to_csv(staging / filename, index=False)

        (staging / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metadata["artifacts"] = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }

        (staging / "receipt.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        staging.rename(output_dir)

    manifest_records = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest_records.append(
                {
                    "artifact": f"recovery_{path.stem}",
                    "stage": "formal_recovery",
                    "timestamp_utc": timestamp,
                    "git_commit": commit,
                    "config_sha256": metadata["config_sha256"],
                    "raw_data_sha256": audit["raw_data_sha256"],
                    "raw_archive_sha256": audit.get("raw_archive_sha256") or "",
                    "raw_source_mode": audit["raw_source_mode"],
                    "processed_data_sha256": audit["processed_data_sha256"],
                    "data_pipeline_sha256": audit["data_pipeline_sha256"],
                    "artifact_sha256": sha256_file(path),
                    "fit_status": metadata["status"],
                    "path": str(path.relative_to(root)),
                }
            )

    append_manifest(
        repository_path(config, config["outputs"]["manifest"]),
        manifest_records,
    )

    print(f"Formal recovery outputs written to {output_dir}")
    if issues:
        raise RuntimeError(
            f"Formal recovery completed with blocking issues: {issues}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        run_smoke(config)
    else:
        run_formal(config, confirmed=args.confirm_design_frozen)


if __name__ == "__main__":
    main()