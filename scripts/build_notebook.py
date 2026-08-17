#!/usr/bin/env python3
"""Regenerate the reader-facing notebook scaffold with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Probability Discounting: Joint Choice and RT Models\n\n"
            "Reproducible companion notebook for the final report. Core calculations live "
            "in `src/pd_project`; this notebook checks inputs and presents finalized outputs."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            "**Scaffold status:** no empirical conclusion is stated until the frozen pipeline "
            "has produced recovery, reliability, and held-out prediction artifacts. Replace "
            "this paragraph only after checking those executed outputs."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The fixed hyperbolic valuation and logistic choice rule are shared by M1-M3. "
            "The models differ only in the value-to-RT predictor and share one log-normal "
            "RT observation family.\n\n"
            "### Key Assumptions\n\n"
            "- Run A is used for inference; run B is used for held-out evaluation.\n"
            "- Choice and RT are conditionally independent given the latent subjective values "
            "and model parameters.\n"
            "- Reward and loss have distinct discount parameters, while choice sensitivity "
            "is shared.\n"
            "- Participant-level scores are averaged within condition before group summaries."
        ),
        nbf.v4.new_markdown_cell("### 1. Setup"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'config' / 'analysis.yaml').exists():\n"
            "    raise RuntimeError('Run this notebook with the repository root as the working directory.')\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
            "print('Repository root detected.')"
        ),
        nbf.v4.new_markdown_cell("### 2. Load the frozen configuration"),
        nbf.v4.new_code_cell(
            "from pd_project.config import load_config\n\n"
            "config = load_config(PROJECT_ROOT / 'config' / 'analysis.yaml')\n"
            "print('Freeze status:', config['project']['freeze_status'])\n"
            "print('Formal run B enabled:', config['project']['formal_run_b_enabled'])\n"
            "print('Enabled primary models:', [m for m in ('M1', 'M2', 'M3') if config['models'][m]['enabled']])"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 3. Check prepared inputs"),
        nbf.v4.new_code_cell(
            "processed_path = PROJECT_ROOT / config['data']['processed_trials']\n"
            "audit_path = PROJECT_ROOT / config['data']['audit_report']\n"
            "if processed_path.exists() and audit_path.exists():\n"
            "    import json\n"
            "    import pandas as pd\n"
            "    trials = pd.read_csv(processed_path)\n"
            "    audit = json.loads(audit_path.read_text(encoding='utf-8'))\n"
            "    print(f\"Loaded {len(trials):,} prepared trials\")\n"
            "    print('Audit passed expected counts:', audit['passed_expected_counts'])\n"
            "else:\n"
            "    trials = None\n"
            "    print('Prepared data are not present. Run scripts/prepare_data.py after adding local .mat files.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Results\n\n"
            "### 4. Locate finalized artifacts\n\n"
            "Do not interpret model winners from this section until both freeze gates are complete."
        ),
        nbf.v4.new_code_cell(
            "expected_artifacts = [\n"
            "    PROJECT_ROOT / 'results' / 'run_a_fits.csv',\n"
            "    PROJECT_ROOT / 'results' / 'formal_run_b' / 'participant_condition_scores.csv',\n"
            "    PROJECT_ROOT / 'results' / 'formal_run_b' / 'run_b_reliability_fits.csv',\n"
            "]\n"
            "for artifact in expected_artifacts:\n"
            "    print(('READY  ' if artifact.exists() else 'MISSING'), artifact.relative_to(PROJECT_ROOT))"
        ),
        nbf.v4.new_markdown_cell(
            "### 5. Required result panels\n\n"
            "Once artifacts exist, add compact, source-backed panels for:\n\n"
            "1. parameter and model recovery;\n"
            "2. run A/B reliability and agreement;\n"
            "3. run-B choice and RT prediction by reward/loss condition;\n"
            "4. full-vs-choice-only differences in choice prediction, recovery, and reliability;\n"
            "5. in-support versus out-of-support diagnostics."
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "Complete this section only after all cells have been rerun from a clean kernel and "
            "the numerical statements have been reconciled with the exported report tables."
        ),
    ]
    output = root / "final_project.ipynb"
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
