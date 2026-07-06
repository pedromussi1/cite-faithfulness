"""Error-bar figures for the controlled study.

Renders a grouped bar chart of citation precision/recall/F1 per configuration
with bootstrap-CI error bars — the visual companion to ``report.py``'s table.

matplotlib is an optional dependency (``pip install -e ".[viz]"``); the
data-shaping step (``figure_series``) is pure Python and unit-tested so the
numbers behind the figure are checkable without a plotting backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .report import METRICS, ConfigSummary


@dataclass(frozen=True)
class Series:
    """One metric's values + asymmetric error-bar half-widths across configs."""

    metric: str
    labels: list[str]
    points: list[float]
    err_lo: list[float]  # point - lo (lower half-width, >= 0)
    err_hi: list[float]  # hi - point (upper half-width, >= 0)


def figure_series(summaries: list[ConfigSummary]) -> list[Series]:
    """Turn config summaries into per-metric plot series with CI half-widths."""
    labels = [s.name for s in summaries]
    series: list[Series] = []
    for m in METRICS:
        points = [s.cis[m].point for s in summaries]
        err_lo = [max(0.0, s.cis[m].point - s.cis[m].lo) for s in summaries]
        err_hi = [max(0.0, s.cis[m].hi - s.cis[m].point) for s in summaries]
        series.append(Series(metric=m, labels=labels, points=points, err_lo=err_lo, err_hi=err_hi))
    return series


def render_figure(summaries: list[ConfigSummary], out_path: Path) -> Path:
    """Render the grouped bar chart with CI error bars to ``out_path`` (PNG)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; no display needed
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without [viz]
        raise SystemExit(
            "matplotlib is required for figures. Install with: pip install -e \".[viz]\""
        ) from exc

    series = figure_series(summaries)
    labels = series[0].labels if series else []
    n_groups = len(labels)
    n_metrics = len(series)
    width = 0.8 / max(1, n_metrics)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_groups), 4.5))
    for i, s in enumerate(series):
        xs = [j + i * width for j in range(n_groups)]
        ax.bar(
            xs,
            s.points,
            width=width,
            yerr=[s.err_lo, s.err_hi],
            capsize=4,
            label=s.metric.replace("citation_", ""),
        )
    ax.set_xticks([j + width * (n_metrics - 1) / 2 for j in range(n_groups)])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Citation faithfulness by configuration (95% bootstrap CIs)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
