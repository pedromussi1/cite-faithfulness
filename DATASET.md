# Evaluation dataset — Week 1 (pinned)

## Decision

**Week 1 uses PaperPal's own hand-curated question set over 4 canonical ML
papers** (Attention Is All You Need, BERT, LoRA, CLIP), not a public ALCE
dataset (ASQA/ELI5). Rationale:

- **Ground truth is trustworthy.** The 26 questions were hand-written with
  human-verified answer pages. ALCE's ASQA/ELI5 sets are large but their
  passages/answers are noisier and the corpora (Wikipedia/web) are a poor
  match for PaperPal, whose whole purpose is question-answering over
  *research PDFs*. Using PaperPal's real corpus keeps the study on-distribution.
- **Zero cost / fully local.** The 4 PDFs are bundled (`data/papers/`); no
  download, no API, no license friction (all are open-access arXiv papers).
- **The system under test is unchanged.** Reusing PaperPal's fixtures means the
  faithfulness numbers describe the *actual shipped system*, not a re-indexed
  approximation.

The **citation metric itself is a faithful reimplementation of ALCE's** (NLI
entailment of the cited passage against the sentence — see `citeval/metrics.py`).
So this is "ALCE metric, PaperPal data," which is the honest framing for the
writeup: we reproduce the *method*, and separately validate it on a public
slice as a stretch goal (below).

## What's pinned

| Asset | Path | Notes |
|---|---|---|
| Papers (4) | `data/papers/*.pdf` | sha256[:16] ids match the question `paper_id`s |
| Questions (26) | `data/questions.jsonl` | copied verbatim from PaperPal `backend/eval/dataset.jsonl` |

`data/questions.jsonl` schema (one JSON object per line):

```json
{"id": "att-01", "category": "explanation",
 "question": "...", "expected_pages": [2],
 "gold_answer": "...", "paper_id": "bdfaa68d8984f0dc"}
```

`expected_pages` / `gold_answer` are **not** needed by the ALCE citation metric
(it is reference-free — it checks passage→sentence entailment). They are kept
for (a) cross-checking against PaperPal's page-overlap metric and (b) the
answer-correctness metric added in Week 2.

## Known limitations (to address later)

- **Size (n=26).** Fine for Week-1 wiring and for bootstrap CIs, but thin.
  **Week 3** expands to ~100–200 hand-audited questions and adds *statement-level
  support labels* so the NLI verifier's own precision/recall can be measured
  against human judgement.
- **LLM-generated vs hand-curated.** PaperPal's eval report already documented
  that model-generated questions are easier than hand-curated ones; the
  expansion set will stay hand-curated for exactly this reason.

## Stretch goal — public-slice reproduction

To claim a clean *reproduction* of an ALCE number, run the metric on a small
ASQA slice with ALCE's released model outputs and confirm our citation
precision/recall lands within tolerance of the paper's reported values. Tracked
as a Week-1/2 stretch task; not required for the controlled study, which only
needs the metric to be *correct* (pinned by `tests/test_metrics.py`).
