# Probability Discounting: Joint Choice and Reaction-Time Models

This repository is the reproducible analysis scaffold for the Computational
Cognitive Science final project. It compares three psychologically distinct
value-to-reaction-time mappings while holding the probability-discounting
valuation model and logistic choice rule fixed.

## Analysis status

- Dataset: Probability Discounting (49 participants; run A and run B).
- Valuation: hyperbolic probability discounting with condition-specific
  `k_R` and `k_L`.
- Choice rule: logistic choice probability based on `delta_v`.
- RT observation model: log-normal with one of three fixed predictors.
- Estimation: participant-level bounded MLE with deterministic multistart.
- Current freeze state: **candidate**. Do not produce formal run-B results
  until both freeze gates in `docs/analysis_freeze.md` are complete.

The repository intentionally contains no participant data and no official
course LaTeX template. Add those locally using the instructions below.

## Model set

| Model | RT predictor | Psychological hypothesis |
|---|---|---|
| M1 | `abs(delta_v)` | Larger value separation makes the decision easier and faster. |
| M2 | `delta_v ** 2` | Large value separation produces a super-linear speed-up. |
| M3 | `(abs(v_cert) + abs(v_uncert)) / 2` | Overall subjective magnitude increases response vigour. |

All models use the same valuation, choice rule, condition structure, and
log-normal RT noise family. The choice-only baseline is fit once with the RT
term removed.

In the report, `delta_v` is called the **drift-rate prior** following the
course convention. It is a deterministic trial signal here, not a Bayesian
prior and not a claim that this factorized model is a full DDM. M3's extension
of absolute magnitude to losses is a hypothesis being tested, not an already
established empirical result.

## Repository layout

```text
config/analysis.yaml       single source of analysis settings
docs/                      model contract, freeze checklist, and decision log
data/raw/                  local MATLAB files (not committed)
data/processed/            generated tidy data (not committed)
src/pd_project/            shared analysis library
scripts/                   command-line pipeline entry points
tests/                     synthetic unit and smoke tests
final_project.ipynb        reader-facing report companion
results/manifest.csv       generated provenance index (template is tracked)
figures/                   generated figures (not committed by default)
report/                    official LaTeX template goes here
disclosures/               per-member contribution/disclosure records
```

## Setup

```bash
conda env create -f environment.yml
conda activate ccs-pd
python -m pip install -e .
python -m pytest -q
```

Alternatively, install with a regular virtual environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

## Add the local inputs

1. Put the original `PD data.zip` in `data/raw/`. The preparation script safely
   extracts it automatically. Supplying the 49 `.mat` files directly is also
   supported.
2. Put the official course LaTeX files in `report/` without changing their
   layout definitions.
3. Check the raw MATLAB keys and the zero-based column positions in
   `config/analysis.yaml` before preparing data.

Raw data, processed participant-level data, optimizer traces, and simulation
outputs are ignored by Git. Never commit identifying or course-restricted
data. Completed member records are also ignored because they may contain names
or matriculation numbers; submit them through the course channel instead.

## Reproducible workflow

### Gate 1: design freeze

Complete and sign `docs/analysis_freeze.md`. Confirm M1-M3, data masks,
metrics, random seeds, recovery design, and the run-B access rule.

### 1. Prepare and audit data

```bash
python scripts/prepare_data.py --config config/analysis.yaml
```

This creates `data/processed/pd_trials.csv` and an audit JSON. The script
checks action encoding, independent choice/RT masks, probability conversion,
odds reconstruction, and expected dataset counts.

### 2. Run synthetic checks

```bash
python -m pytest -q
python scripts/run_recovery.py --config config/analysis.yaml --smoke
```

CI runs only synthetic tests; it never accesses run B or private data.

### 3. Validate recovery and numerical settings

Use run A plus synthetic recovery to resolve parameter bounds and any global
MLE-to-MAP fallback decision. Record changes in `docs/decision_log.md`.
The repository currently implements the simulate-refit kernel and recovery
summaries, but not the complete 200-participant parameter/model-recovery
orchestration. Implement and validate that path before changing its readiness
flags.

### Gate 2: execution freeze

Change `project.freeze_status` to `frozen`, set
`project.formal_run_b_enabled` to `true`, record the config hash and Git
commit, set every `pipeline_readiness` flag to `true`, resolve the RT
sensitivity cutoffs, and obtain all group-member approvals.

### 4. Fit run A

```bash
python scripts/fit_run_a.py --config config/analysis.yaml
```

### 5. Formal run-B evaluation

```bash
python scripts/run_b_once.py --config config/analysis.yaml
```

The run-B command refuses to run unless the tracked registry says `not_run`.
It changes `config/formal_run_b_status.json` to `in_progress` before accessing
formal results and to `completed` only after success. After a successful run,
commit and push that registry immediately so other clones cannot start a
second formal evaluation. Any interrupted run requires a documented manual
audit; the script does not silently resume or reset the registry.

Because the local lock cannot coordinate separate clones, designate exactly
one formal-run operator and obtain written confirmation that all other members
have stopped their copies before execution. The corresponding readiness flag
must remain false until this coordination and private artifact-archive policy
have been agreed.

### 6. Build report outputs

```bash
python scripts/make_outputs.py --config config/analysis.yaml
jupyter nbconvert --execute --to notebook --inplace final_project.ipynb
```

The current notebook is an executable scaffold and presentation layer. Before
submission it must become the top-level orchestrator that can rebuild all
reported figures and numbers from the local raw archive without manual edits,
while reusing a matching completed one-shot run-B fingerprint rather than
creating a second formal evaluation. The `notebook_end_to_end_implemented`
readiness flag intentionally blocks Gate 2 until that work is complete.

## Submission checks

Before submission, verify all of the following:

- compilable official LaTeX source, compiled PDF, and all analysis code are
  present;
- each member has a separate disclosure/responsibility record;
- the official `macros_FP` layout remains A4, 11 pt, single-column;
- Abstract is at most 250 words;
- Introduction through Discussion is at most five pages;
- figures, tables, and references appear after the main body;
- required section/subsection headings are unchanged and all red template
  prompts are removed;
- member contributions and requested assessment type are stated;
- a clean notebook run reproduces every reported table, figure, and number.

## Collaboration conventions

- One shared implementation owns valuation, choice, likelihood, and data
  handling.
- Each member may own one RT model's full recovery/reliability/prediction
  analysis, but should not copy shared code.
- Use short feature branches and pull requests; require the test workflow to
  pass before merging.
- Log every frozen decision and every generated artifact's provenance.
