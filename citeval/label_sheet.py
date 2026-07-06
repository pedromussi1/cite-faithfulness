"""Generate a human-labeling sheet from a run.

The verifier's detection metric in `verify_eval` uses a *proxy* for ground
truth (gold-page hit). The definitive version needs a human to judge, for each
cited sentence, whether the cited passage actually supports the claim. This
emits exactly that sheet: one row per cited sentence with the claim, the pages
it cites, the cited passage text, and the NLI verdict — plus an empty
``human_supported`` field to fill in (``yes``/``no``).

Usage:

    python -m citeval.label_sheet --run dense-8b --out labels/dense-8b.jsonl

Label the ``human_supported`` field, then those labels can replace the
gold-page proxy for a definitive detection precision/recall (Week 3+).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import build_passage_map, parse_citations, split_sentences, strip_citations
from .nli import get_nli
from .report import RUNS_DIR, load_run


def build_sheet(run: str, nli_name: str, threshold: float) -> list[dict]:
    nli = get_nli(nli_name, **({} if nli_name == "mock" else {"threshold": threshold}))
    rows = [r for r in load_run(run) if r.get("retrieved") and r.get("answer")]
    sheet: list[dict] = []
    for row in rows:
        passages = build_passage_map(row["retrieved"])
        for sent in split_sentences(row["answer"]):
            keys = parse_citations(sent)
            if not keys:
                continue
            hyp = strip_citations(sent)
            if not hyp:
                continue
            cited_text = "\n\n".join(t for k in keys if (t := passages.get(k, "")))
            sheet.append(
                {
                    "run": run,
                    "id": row["id"],
                    "claim": hyp,
                    "cited_pages": [f"{p}:{pg}" for p, pg in keys],
                    "cited_passage": cited_text,
                    "gold_pages": row.get("expected_pages", []),
                    "nli_supported": bool(cited_text) and nli.entails(cited_text, hyp),
                    "human_supported": "",  # <- fill in: "yes" / "no"
                }
            )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sheet = build_sheet(args.run, args.nli, args.threshold)
    out = Path(args.out) if args.out else RUNS_DIR.parent / "labels" / f"{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in sheet) + "\n", encoding="utf-8")
    print(f"Wrote {len(sheet)} cited-sentence rows to {out}")
    print("Fill in each row's 'human_supported' (yes/no) to enable the definitive eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
