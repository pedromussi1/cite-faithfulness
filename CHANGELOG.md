# Changelog

All notable changes to **cite-faithfulness** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
Minor versions track the project's weekly research milestones (see the
[project plan](../)); `v1.0.0` marks the completed study + writeup.

## [Unreleased]

### Planned (Week 3 — novel verifier)
- A lightweight NLI citation verifier (flag + filter/rerank modes) vs. a
  self-consistency baseline, measured against human labels.
- Dataset expansion to ~100–200 hand-audited questions with statement-level
  support labels.

## [0.2.3] — 2026-07-06

### Fixed
- **Runaway-generation hang** (`run_faithfulness`): a small local model can
  enter a token-repetition loop and stream forever, which the per-read HTTP
  timeout never catches — hanging the whole sweep on one question. Added a hard
  per-question cap (`--query-timeout`, default 120s): a slower answer is
  abandoned, logged as a timeout error, and the sweep continues.

### Added
- **Resumable sweep** (`scripts/run_sweep.ps1`): cells that already produced a
  `summary.json` are skipped by default, so an interrupted sweep resumes where
  it stopped instead of redoing completed cells. `-Force` re-runs everything.

## [0.2.2] — 2026-07-06

### Added
- **Offline re-scoring** (`citeval/rescore.py`): each eval run now stores the
  full retrieved passages, so any change to the NLI judge, threshold, or metric
  can be applied to already-collected answers in **seconds** — no PaperPal, no
  Ollama. Generating answers (slow, GPU) is now decoupled from scoring (cheap,
  CPU). `python -m citeval.rescore --all --in-place` re-scores every run.
- `make rescore` target; README "iterate on scoring" section.

### Changed
- `run_faithfulness` result rows now include a `retrieved` field (passages with
  text) alongside the existing `retrieved_keys`.

## [0.2.1] — 2026-07-06

### Fixed
- **NLI passage truncation** (`citeval/nli.py`): the cross-encoder judge
  truncates at 512 tokens, so a supporting fact near the *end* of a long
  retrieved passage was silently cut and misjudged as unsupported — artificially
  depressing citation recall. `CrossEncoderNLI` now splits long premises into
  overlapping windows (`pack_windows`), scores the hypothesis against each, and
  takes the max entailment, making the verdict independent of where in the
  passage the support lives. Verified: a fact buried after 6k chars of filler
  went from entail=0.006 (miss) to 0.997 (correct); unrelated long text stays
  0.0 (no false positives). Added `entail_prob()` and 5 windowing unit tests.

## [0.2.0] — 2026-07-06

Second milestone: the controlled study — differences between configurations
are now reported *with uncertainty*, not as bare point estimates.

### Added
- **Statistics module** (`citeval/stats.py`, pure stdlib, unit-tested):
  percentile **bootstrap 95% confidence intervals**, a **paired bootstrap**
  test for the mean F1 difference between two configs (CI + two-sided
  p-value), and an **exact McNemar test** for the per-question
  fully-supported outcome. Every result is reproducible via `--seed`.
- **Results report** (`citeval/report.py`): aggregates any set of runs into a
  `REPORT.md` with per-config precision/recall/F1 + CIs and a pairwise
  significance table vs. a chosen baseline. A ΔF1 CI spanning zero is reported
  as an honest non-result.
- **Config-sweep orchestrator** (`scripts/run_sweep.ps1`): runs the retrieval
  config (dense / hybrid / rerank / hybrid+rerank) × model size
  (Llama 3.2 3B vs 3.1 8B) factorial, restarting PaperPal per cell with the
  right env vars — mirrors PaperPal's own `run_ablation.ps1`.
- **Error-bar figures** (`citeval/plots.py`, optional `[viz]` extra): grouped
  bar chart of the metrics with bootstrap-CI error bars.
- 13 new unit tests (stats correctness + report over synthetic runs).

### Notes
- The metric and statistics are verified offline; producing the live results
  table requires running the sweep against PaperPal + Ollama on a GPU host.

## [0.1.0] — 2026-07-06

First milestone: the faithfulness metric, reproduced and tested, wired to the
live PaperPal RAG system, with the Week-1 dataset pinned.

### Added
- **ALCE citation faithfulness metric** (`citeval/metrics.py`) — a faithful
  reimplementation of the citation precision/recall definitions from Gao et
  al., 2023 (*Enabling LLMs to Generate Text with Citations*, EMNLP), including
  the official "necessary-member" precision rule, per-sentence support scoring,
  and hallucinated-citation detection (citations to never-retrieved passages).
- **Pluggable NLI backends** (`citeval/nli.py`): `CrossEncoderNLI` (local
  DeBERTa-MNLI cross-encoder, $0, runs on CPU/GPU) for real runs, and a
  deterministic `KeywordNLI` stub for tests/CI with no model download.
- **PaperPal client + eval driver** (`citeval/client.py`,
  `citeval/run_faithfulness.py`): drives a running PaperPal backend over its
  `/query` SSE API and archives per-question results + `summary.json`.
- **Self-contained demo** (`citeval/demo.py`) and **9 unit tests** pinning the
  metric logic — both run with zero downloads and no server.
- **Pinned Week-1 dataset**: PaperPal's 4 fixture papers (Attention, BERT,
  LoRA, CLIP) + 26 hand-curated questions (`data/`, documented in `DATASET.md`).
- Project scaffolding: `pyproject.toml`, `Makefile`, MIT `LICENSE`, GitHub
  Actions CI (ruff + pytest + demo), README.

### Notes
- This release establishes the *method* and *harness*. Live faithfulness
  numbers against Ollama models, the controlled study, and the NLI verifier
  follow in subsequent milestones.

[Unreleased]: https://github.com/pedromussi1/cite-faithfulness/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/pedromussi1/cite-faithfulness/releases/tag/v0.2.3
[0.2.2]: https://github.com/pedromussi1/cite-faithfulness/releases/tag/v0.2.2
[0.2.1]: https://github.com/pedromussi1/cite-faithfulness/releases/tag/v0.2.1
[0.2.0]: https://github.com/pedromussi1/cite-faithfulness/releases/tag/v0.2.0
[0.1.0]: https://github.com/pedromussi1/cite-faithfulness/releases/tag/v0.1.0
