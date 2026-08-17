"""Plotting helpers that consume finalized result tables only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_project_style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300})


def save_figure(figure: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    return output


def predictive_score_panel(
    summary: pd.DataFrame,
    *,
    score: str,
    ylabel: str,
) -> Any:
    """Create a compact model-by-condition point plot from pre-aggregated scores."""

    required = {"model", "condition", score}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Predictive summary is missing columns: {missing}")
    set_project_style()
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    sns.pointplot(data=summary, x="model", y=score, hue="condition", errorbar="ci", ax=axis)
    axis.set_xlabel("RT model")
    axis.set_ylabel(ylabel)
    sns.despine(ax=axis)
    return figure

