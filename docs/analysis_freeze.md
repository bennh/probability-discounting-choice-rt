# Analysis freeze checklist

Complete both gates before the formal run-B command is enabled. Checkboxes are
project decisions, not results to be filled in after seeing held-out scores.

## Gate 1 - design freeze

- [ ] Dataset fixed to Probability Discounting.
- [ ] M1, M2, and M3 formulas and psychological hypotheses approved.
- [ ] M4 explicitly excluded. If the group wants it, implement its public API,
      bounds, simulation, scoring, and tests before returning to Gate 1.
- [ ] Hyperbolic valuation, condition-specific `k_R`/`k_L`, and shared `beta`
      approved.
- [ ] Raw action encoding and independent choice/RT masks approved.
- [ ] Probability, odds, amount-scale, and RT-sensitivity rules approved.
- [ ] Primary/secondary metrics and participant-first aggregation approved.
- [ ] Recovery, reliability, bootstrap, multiplicity, and support-shift plans
      approved.
- [ ] Master seed and deterministic seed-keying scheme approved.
- [ ] Formal run-B outputs embargoed until Gate 2.
- [ ] Responsibilities assigned and recorded below.

## Gate 2 - execution freeze

- [ ] Data audit passes and deviations from expected counts are resolved.
- [ ] Unit tests and synthetic smoke recovery pass.
- [ ] Complete parameter recovery and model-recovery confusion matrix reviewed.
- [ ] One-shot run-B path smoke-tested on synthetic data, including interruption
      and registry behavior.
- [ ] Parameter bounds are resolved using run A and recovery only.
- [ ] MLE or global MAP-fallback decision is final.
- [ ] Recovery ranges, repetitions, and stress set are final.
- [ ] Model-recovery rule and all comparison families are final.
- [ ] `config/analysis.yaml` status changed to `frozen`.
- [ ] `formal_run_b_enabled` changed to `true`.
- [ ] Every `pipeline_readiness` flag changed to `true` only after its
      implementations and tests are complete.
- [ ] Projected-gradient convergence threshold calibrated in the locked
      Python/NumPy/SciPy environment and frozen as a positive number.
- [ ] Exactly one formal-run operator designated; all other clones stopped.
- [ ] Private archival location and integrity-check procedure for formal
      artifacts approved.
- [ ] Final `s0` assertion and resolved RT sensitivity cutoffs match the audit.
- [ ] ICC definition, participant-cluster bootstrap, comparison families, and
      support-shift outputs are frozen.
- [ ] Merten & Kollau (2026) source/BibTeX obtained from the teacher or LMS.
- [ ] Config, Git, raw-data, processed-data, and run-A-fit fingerprints recorded.
- [ ] Individual contribution/disclosure records prepared.
- [ ] All members approve the exact frozen configuration.

## Sign-off

| Member | Responsibility | Gate 1 date | Gate 2 date | Approval |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

## Frozen fingerprints

| Item | Value |
|---|---|
| Freeze version | |
| Config SHA256 | |
| Git commit | |
| Raw-data SHA256 | |
| Processed-data SHA256 | |
| Run-A fits SHA256 | |
