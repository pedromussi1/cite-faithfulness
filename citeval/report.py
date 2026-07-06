"""Aggregate eval runs into a results report with confidence intervals.

Reads one or more ``runs/<name>/results.jsonl`` files (produced by
``citeval.run_faithfulness``), and for each configuration computes citation
precision / recall / F1 as a point estimate with a bootstrap 95% CI. When a
baseline is given, it adds a pairwise-significance table: the paired bootstrap
difference in F1 (CI + p-value) and an exact McNemar test on the per-question
"fully supported" outcome.

Usage (from repo root):

    python -m citeval.report --runs w2-dense w2-hybrid w2-rerank --baseline w2-dense
    python -m citeval.report --all --baseline w2-dense-3b    # every run under runs/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stats import CI, DiffResult, McNemarResult, bootstrap_ci, mcnemar_exact, paired_diff

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Per-question metric keys inside each row's "faithfulness" object.
METRICS = ("citation_precision", "citation_recall", "citation_f1")


def load_run(name_or_path: str) -> list[dict[str, Any]]:
    """Load a run's non-error result rows. Accepts a run name or a path."""
    p = Path(name_or_path)
    if p.is_dir():
        p = p / "results.jsonl"
    elif not p.exists():
        p = RUNS_DIR / name_or_path / "results.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Run not found: {name_or_path} (looked at {p})")
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row or "faithfulness" not in row:
            continue
        rows.append(row)
    return rows


def metric_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Map question id -> metric value for one metric key."""
    return {row["id"]: float(row["faithfulness"][key]) for row in rows}


def supported_by_id(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Per-question binary: 1 if the answer was fully citation-supported
    (recall == 1.0), else 0. Used for the McNemar test."""
    return {row["id"]: int(float(row["faithfulness"]["citation_recall"]) >= 1.0) for row in rows}


@dataclass(frozen=True)
class ConfigSummary:
    name: str
    n: int
    cis: dict[str, CI]  # metric key -> CI
    supported_rate: float
    rows: list[dict[str, Any]] = field(default_factory=list)


def summarize(name: str, rows: list[dict[str, Any]], *, n_boot: int, seed: int) -> ConfigSummary:
    cis = {
        m: bootstrap_ci([float(r["faithfulness"][m]) for r in rows], n_boot=n_boot, seed=seed)
        for m in METRICS
    }
    supp = supported_by_id(rows)
    rate = sum(supp.values()) / len(supp) if supp else 0.0
    return ConfigSummary(name=name, n=len(rows), cis=cis, supported_rate=rate, rows=rows)


def _align(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float]]:
    """Align two id->value maps on their common ids (sorted for determinism)."""
    common = sorted(set(a) & set(b))
    return [a[i] for i in common], [b[i] for i in common]


@dataclass(frozen=True)
class Pairwise:
    config: str
    baseline: str
    f1_diff: DiffResult
    mcnemar: McNemarResult
    n_paired: int


def compare(
    config: ConfigSummary,
    baseline: ConfigSummary,
    *,
    n_boot: int,
    seed: int,
) -> Pairwise:
    a_f1, b_f1 = _align(
        metric_by_id(config.rows, "citation_f1"),
        metric_by_id(baseline.rows, "citation_f1"),
    )
    diff = paired_diff(a_f1, b_f1, n_boot=n_boot, seed=seed)
    a_s, b_s = _align(supported_by_id(config.rows), supported_by_id(baseline.rows))
    mc = mcnemar_exact([int(x) for x in a_s], [int(x) for x in b_s])
    return Pairwise(config.name, baseline.name, diff, mc, len(a_f1))


def render_report(
    summaries: list[ConfigSummary],
    baseline: ConfigSummary | None,
    *,
    n_boot: int,
    seed: int,
) -> str:
    lines: list[str] = [
        "# Faithfulness controlled study — results",
        "",
        f"_Bootstrap 95% CIs ({n_boot:,} resamples, seed={seed}). "
        "Citation precision/recall follow the ALCE (Gao et al., 2023) definitions._",
        "",
        "## Per-configuration citation faithfulness",
        "",
        "| Config | n | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Fully-supported |",
        "|---|---:|---|---|---|---:|",
    ]
    for s in summaries:
        p, r, f = s.cis["citation_precision"], s.cis["citation_recall"], s.cis["citation_f1"]
        lines.append(
            f"| `{s.name}` | {s.n} "
            f"| {p.point:.3f} [{p.lo:.3f}, {p.hi:.3f}] "
            f"| {r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}] "
            f"| {f.point:.3f} [{f.lo:.3f}, {f.hi:.3f}] "
            f"| {s.supported_rate:.0%} |"
        )

    if baseline is not None:
        lines += [
            "",
            f"## Pairwise significance vs. baseline `{baseline.name}`",
            "",
            "Paired bootstrap on citation F1 (same questions), and an exact "
            "McNemar test on the per-question fully-supported outcome. "
            "`*` marks a 95% CI that excludes zero.",
            "",
            "| Config | ΔF1 vs baseline (95% CI) | bootstrap p | McNemar p (supported) |",
            "|---|---|---:|---:|",
        ]
        for s in summaries:
            if s.name == baseline.name:
                continue
            cmp = compare(s, baseline, n_boot=n_boot, seed=seed)
            d = cmp.f1_diff
            star = " *" if d.significant else ""
            lines.append(
                f"| `{s.name}` | {d.diff:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]{star} "
                f"| {d.p_value:.4f} | {cmp.mcnemar.p_value:.4f} |"
            )

    lines += [
        "",
        "## How to read this",
        "",
        "- A CI that spans zero in the ΔF1 column means the difference from the "
        "baseline is **not** statistically distinguishable at this sample size — "
        "an honest negative result, not a win.",
        "- Precision/recall trade off: reranking often raises precision while a "
        "wider candidate pool raises recall. F1 is the headline.",
        "- Small n (see the `n` column) yields wide CIs; the Week-3 dataset "
        "expansion tightens them.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    # Keep Unicode (Δ, ≥) in the Markdown file, but don't let a legacy Windows
    # console codepage (cp1252) crash the status print.
    import contextlib
    import sys

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="Run names (subdirs under runs/).")
    parser.add_argument("--all", action="store_true", help="Use every run under runs/.")
    parser.add_argument("--baseline", default=None, help="Baseline run name for pairwise tests.")
    parser.add_argument("--out", default=str(RUNS_DIR / "REPORT.md"))
    parser.add_argument("--figures", action="store_true", help="Also render error-bar PNG(s).")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_names = args.runs
    if args.all:
        run_names = sorted(
            p.parent.name for p in RUNS_DIR.glob("*/results.jsonl")
        )
    if not run_names:
        print("No runs specified. Use --runs <names...> or --all.")
        return 2

    summaries = [
        summarize(name, load_run(name), n_boot=args.n_boot, seed=args.seed) for name in run_names
    ]
    baseline = next((s for s in summaries if s.name == args.baseline), None)
    if args.baseline and baseline is None:
        print(f"Baseline '{args.baseline}' is not among the runs.")
        return 2

    report = render_report(summaries, baseline, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {out}")

    if args.figures:
        from .plots import render_figure

        fig_path = render_figure(summaries, out.parent / "figures" / "faithfulness.png")
        print(f"Figure written to {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
