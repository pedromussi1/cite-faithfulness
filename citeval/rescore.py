"""Re-score an existing run offline — no PaperPal, no Ollama.

Because each result row stores the full retrieved passages (see
``run_faithfulness``), any change to the NLI judge or the metric can be applied
to already-collected answers in seconds instead of re-querying the models. This
is what keeps scoring cheap to iterate: generate answers once (slow, needs GPU),
re-score many times (fast, CPU-only).

Usage (from repo root):

    # write a re-scored copy alongside the original
    python -m citeval.rescore --run dense-8b --nli cross-encoder/nli-deberta-v3-base

    # overwrite the run in place
    python -m citeval.rescore --run dense-8b --in-place --nli cross-encoder/...

    # re-score every run under runs/
    python -m citeval.rescore --all --nli cross-encoder/...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .metrics import score_answer
from .nli import get_nli
from .run_faithfulness import RUNS_DIR, summarize

# Fields carried over unchanged from the original row (everything except the
# scoring outputs, which we recompute).
_SCORE_KEYS = {"faithfulness", "sentences", "ts"}


def rescore_row(row: dict[str, Any], nli) -> dict[str, Any] | None:
    """Recompute a row's faithfulness from its stored answer + retrieved.

    Returns None if the row can't be re-scored (an error row, or an older run
    saved before passage text was stored — those need a fresh eval run).
    """
    if "error" in row:
        return row
    if "answer" not in row or "retrieved" not in row:
        return None
    score = score_answer(row["answer"], row["retrieved"], nli)
    base = {k: v for k, v in row.items() if k not in _SCORE_KEYS}
    return {
        **base,
        "faithfulness": {
            "citation_precision": score.citation_precision,
            "citation_recall": score.citation_recall,
            "citation_f1": score.citation_f1,
            "n_sentences": score.n_sentences,
            "n_cited_sentences": score.n_cited_sentences,
            "n_citations": score.n_citations,
            "n_hallucinated": score.n_hallucinated,
        },
        "sentences": [asdict(s) for s in score.sentences],
        "rescored_ts": datetime.now(UTC).isoformat(),
    }


def rescore_run(run: str, out_name: str, nli_name: str, threshold: float) -> int:
    src = RUNS_DIR / run / "results.jsonl"
    if not src.exists():
        print(f"Run not found: {src}")
        return 2
    nli = get_nli(nli_name, **({} if nli_name == "mock" else {"threshold": threshold}))

    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    rescored: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        new = rescore_row(row, nli)
        if new is None:
            skipped += 1
            continue
        rescored.append(new)

    if skipped:
        print(
            f"  {skipped} row(s) lack stored passages (pre-v0.2.2 run) — "
            f"re-run the eval to capture them."
        )

    out_dir = RUNS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rescored) + "\n", encoding="utf-8"
    )
    scored = [r for r in rescored if "faithfulness" in r]
    if scored:
        summary = summarize(scored, out_name, nli_name)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"  {run} -> {out_name}: P={summary['citation_precision']:.3f} "
            f"R={summary['citation_recall']:.3f} F1={summary['citation_f1']:.3f} "
            f"(n={summary['n_questions']})"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run name under runs/ to re-score.")
    parser.add_argument("--all", action="store_true", help="Re-score every run under runs/.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the source run.")
    parser.add_argument("--suffix", default="-rescored", help="Suffix for the output run name.")
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.all:
        runs = sorted(p.parent.name for p in RUNS_DIR.glob("*/results.jsonl"))
    elif args.run:
        runs = [args.run]
    else:
        print("Specify --run <name> or --all.")
        return 2

    rc = 0
    for run in runs:
        if run.endswith(args.suffix):
            continue  # don't re-score our own outputs
        out_name = run if args.in_place else f"{run}{args.suffix}"
        rc |= rescore_run(run, out_name, args.nli, args.threshold)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
