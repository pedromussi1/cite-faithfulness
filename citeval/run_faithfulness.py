"""Offline faithfulness eval driver.

For each question in the dataset: query a running PaperPal backend, take the
answer + retrieved passages off the SSE stream, score citation faithfulness
with an NLI model, and archive per-question results plus an aggregate summary
under ``runs/<name>/``.

Usage (from repo root, with PaperPal's uvicorn running):

    # real run — downloads the NLI model once, drives the live backend
    python -m citeval.run_faithfulness --name w1-dense --nli cross-encoder/nli-deberta-v3-base

    # smoke — mock NLI, still needs the backend up, but no model download
    python -m citeval.run_faithfulness --name smoke --nli mock

No backend? See ``python -m citeval.demo`` for a fully self-contained run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import PaperPalClient
from .metrics import score_answer
from .nli import get_nli

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = DATA_DIR / "papers"
DATASET = DATA_DIR / "questions.jsonl"
RUNS_DIR = REPO_ROOT / "runs"


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def main(base_url: str, run_name: str, nli_name: str, threshold: float) -> int:
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return 2
    questions = load_questions(DATASET)
    print(f"Loaded {len(questions)} questions from {DATASET.name}")

    nli_kwargs = {} if nli_name == "mock" else {"threshold": threshold}
    nli = get_nli(nli_name, **nli_kwargs)

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    per_question: list[dict[str, Any]] = []
    async with PaperPalClient(base_url) as client:
        if not await client.healthz():
            print(f"PaperPal backend not reachable at {base_url}. Start uvicorn first.")
            return 3
        if FIXTURES_DIR.exists():
            mapping = await client.ensure_uploaded(FIXTURES_DIR)
            print(f"Papers in index: {list(mapping.keys())}")

        with results_path.open("w", encoding="utf-8") as out:
            for i, q in enumerate(questions, 1):
                scope = [q["paper_id"]] if q.get("paper_id") else None
                print(f"[{i:>2}/{len(questions)}] {q['id']}: {q['question'][:55]}...", flush=True)
                try:
                    outcome = await client.query(q["question"], paper_ids=scope)
                except Exception as exc:  # noqa: BLE001 — archive and continue
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    out.write(json.dumps({**q, "error": str(exc)}) + "\n")
                    out.flush()
                    continue

                score = score_answer(outcome.answer, outcome.retrieved, nli)
                rec = {
                    **q,
                    "answer": outcome.answer,
                    # Full retrieved payload (incl. passage text) is stored so the
                    # run can be re-scored offline after any NLI/metric change —
                    # see citeval.rescore. This is what makes scoring cheap to
                    # iterate without re-querying PaperPal + Ollama.
                    "retrieved": outcome.retrieved,
                    "retrieved_keys": sorted(
                        {f"{r['paper_id']}:{r['page']}" for r in outcome.retrieved}
                    ),
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
                    "ts": datetime.now(UTC).isoformat(),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                per_question.append(rec)
                f = rec["faithfulness"]
                print(
                    f"  prec={f['citation_precision']:.2f} rec={f['citation_recall']:.2f} "
                    f"f1={f['citation_f1']:.2f} halluc={f['n_hallucinated']}"
                )

    if not per_question:
        print("No successful results.")
        return 1

    summary = summarize(per_question, run_name, nli_name)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"=== Faithfulness summary ({summary['n_questions']} questions, nli={nli_name}) ===")
    print(f"  citation precision : {summary['citation_precision']:.3f}")
    print(f"  citation recall    : {summary['citation_recall']:.3f}")
    print(f"  citation F1        : {summary['citation_f1']:.3f}")
    print(f"  hallucinated cites : {summary['total_hallucinated']}")
    print(f"  results            : {results_path}")
    return 0


def summarize(rows: list[dict[str, Any]], run_name: str, nli_name: str) -> dict[str, Any]:
    n = len(rows)
    mean = lambda key: sum(r["faithfulness"][key] for r in rows) / n  # noqa: E731
    return {
        "run": run_name,
        "nli": nli_name,
        "n_questions": n,
        "citation_precision": mean("citation_precision"),
        "citation_recall": mean("citation_recall"),
        "citation_f1": mean("citation_f1"),
        "total_citations": sum(r["faithfulness"]["n_citations"] for r in rows),
        "total_hallucinated": sum(r["faithfulness"]["n_hallucinated"] for r in rows),
        "ts": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--name", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--nli",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI backend: 'mock' or an HF cross-encoder NLI checkpoint name.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.base_url, args.name, args.nli, args.threshold)))
