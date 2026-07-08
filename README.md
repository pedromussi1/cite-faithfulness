# cite-faithfulness

**Are the citations real?** A study of citation *faithfulness* in
citation-grounded RAG — reproducing the ALCE citation precision/recall metric
(Gao et al., EMNLP 2023) and applying it to a live RAG system
([PaperPal](https://github.com/pedromussi1/PaperPal)).

> **The gap this closes.** A RAG system that cites `[paper_id:page]` can point
> at the *right page* and still say something that page does not support.
> PaperPal's built-in eval measures page-number overlap — a lexical proxy. This
> project measures **whether each cited passage actually entails the sentence it
> is attached to**, using a Natural Language Inference model. That is the
> difference between "cited the right place" and "the citation is faithful."

Status: **Week 3** — metric reproduced (Week 1), controlled study with bootstrap
CIs + significance tests (Week 2), and now the **novel contribution**: an NLI
**citation verifier** that flags unsupported citations and re-attributes them to
the best-supporting retrieved passage — evaluated against human-curated gold
pages so the result is *not* circular. See [`PLAN` in the parent repo](../).

## The metric (ALCE, reproduced)

For each sentence `s` in an answer, with its cited passages `C = {c_1,…,c_k}`:

- **Citation recall** = 1 if the concatenation of `C` *entails* `s` (fully
  supported), else 0. Uncited sentences score 0.
- **Citation precision** = fraction of individual citations that are *not
  irrelevant*: a citation is credited if it alone entails `s`, **or** it is a
  necessary member of a group that jointly entails `s`. Redundant or
  unsupported (e.g. hallucinated) citations are not credited.

Entailment is judged by a pluggable NLI backend (`citeval/nli.py`):
a local DeBERTa-MNLI cross-encoder for real runs, and a deterministic keyword
stub for tests. Full definitions and the "necessary member" rule are documented
in [`citeval/metrics.py`](citeval/metrics.py).

## Quickstart

```bash
pip install -e ".[dev]"          # metric + tests, no torch needed
make demo                        # self-contained ALCE scoring — no server, no downloads
make test                        # unit tests pinning the metric logic
```

`make demo` output (KeywordNLI stub — illustrative numbers, not real NLI):

```
citation precision : 0.500
citation recall    : 0.500
sentences (recall denom) : 4
hallucinated citations   : 1
```

## Running against PaperPal (real NLI)

1. Start PaperPal's backend (`uvicorn app.main:app` in `PaperPal/backend`, with
   Ollama running). It exposes `/query` as an SSE stream.
2. Install the NLI judge and run the eval:

```bash
pip install -e ".[nli]"
python -m citeval.run_faithfulness --name w1-default \
    --nli cross-encoder/nli-deberta-v3-base
```

The driver uploads the 4 bundled papers if missing, asks each of the 26
questions, scores citation faithfulness, and writes per-question rows +
`summary.json` under `runs/<name>/`. `--nli mock` gives a no-download smoke test
(still needs the backend up).

## Controlled study (Week 2)

Sweep the retrieval configuration (`dense` / `hybrid` / `rerank` /
`hybrid+rerank`) against model size (Llama 3.2 3B vs 3.1 8B), then aggregate
with uncertainty:

```bash
# 1. Run the 4×2 sweep — restarts PaperPal per cell with the right env vars.
#    (Windows/PowerShell; mirrors PaperPal's own run_ablation.ps1.)
powershell -File scripts/run_sweep.ps1

# 2. Build the results report: bootstrap 95% CIs + paired significance tests.
python -m citeval.report --all --baseline dense-8b

# 3. Optional error-bar figures.
pip install -e ".[viz]"
python -m citeval.report --all --baseline dense-8b --figures
```

The report gives each configuration's citation precision/recall/F1 as a point
estimate **with a bootstrap 95% CI**, and a pairwise table with the **paired
bootstrap ΔF1 (CI + p-value)** and an **exact McNemar test** on the
per-question fully-supported outcome. A ΔF1 CI that spans zero is reported as an
honest non-result, not a win. The statistics (`citeval/stats.py`) are pure
stdlib and unit-tested; `--seed` makes every interval reproducible.

**Iterate on scoring without re-running the models.** Each run stores the full
retrieved passages, so after any change to the NLI judge, threshold, or metric
you can re-score every archived run in seconds — no PaperPal, no Ollama:

```bash
python -m citeval.rescore --all --in-place    # re-score, then re-report
python -m citeval.report --all --baseline dense-8b
```

Generating answers is the slow part (needs the GPU host); scoring is cheap and
now decoupled from it.

## The citation verifier (Week 3, the novel contribution)

`citeval/verifier.py` applies the NLI judge to a model's citations to:

- **flag** cited sentences whose own citations don't entail the claim, and
- **repair** them by re-attributing the citation to the passage in the
  retrieved pool that best entails the claim (or *abstain* if none does).

```bash
python -m citeval.verify_eval --all          # → runs/VERIFIER_REPORT.md
```

**Avoiding circularity.** Repair uses NLI, so scoring the repaired answer with
the *same* NLI would be trivially perfect. The evaluation instead uses the
**human-curated gold pages** (`expected_pages`) as independent ground truth: it
asks whether the verifier re-attributes citations onto a gold page *more often
than the raw model*, reported as a paired bootstrap ΔF1 with a CI + p-value. For
the definitive detection metric, `citeval/label_sheet.py` emits a
human-labeling sheet (one row per cited sentence) so a person — not the NLI —
judges support:

```bash
python -m citeval.label_sheet --run dense-8b   # → labels/dense-8b.jsonl to fill in
```

## Threshold sensitivity (Week 4)

The verifier's one knob — the entailment threshold `tau` — trades **detection** (flagging
unsupported citations) against **abstention** (declining to re-attribute when nothing entails).
Since both decisions are thresholds over continuous NLI scores that don't depend on `tau`,
`citeval/threshold.py` caches those scores once per run, then sweeps `tau` analytically — so the
whole curve costs a single NLI pass and re-sweeps are instant and torch-free:

```bash
python -m citeval.threshold --all      # score once → runs/THRESHOLD_REPORT.md
python -m citeval.threshold --all --from-cache   # re-sweep, no model load
```

Findings ([`runs/THRESHOLD_REPORT.md`](runs/THRESHOLD_REPORT.md)): detection F1 is nearly flat in
`tau` (the flagger needs no tuning), while repair only worsens as `tau` rises — its Δ vs. the raw
model is never significantly positive and turns significantly negative at high `tau`. Reinforces
*flag, don't auto-repair*.

## Paper

[`PAPER.md`](PAPER.md) is the full write-up — abstract, related work (ALCE / RAGAS / FactScore),
method, results with confidence intervals, and limitations — pulling the study, the verifier, and
the threshold analysis into one narrative.

## Layout

```
citeval/
  metrics.py           ALCE citation precision/recall/F1 (the reproduction)
  nli.py               NLI backends: CrossEncoderNLI (real) + KeywordNLI (tests)
  client.py            async PaperPal /query SSE client
  run_faithfulness.py  offline eval driver → runs/<name>/
  stats.py             bootstrap CIs + paired bootstrap + exact McNemar
  report.py            aggregate runs → REPORT.md (CIs + significance)
  rescore.py           re-score archived runs offline (no server/model)
  verifier.py          NLI citation verifier: flag unsupported + repair citations
  verify_eval.py       evaluate the verifier vs gold pages (non-circular)
  threshold.py         verifier threshold sweep (cached scores → THRESHOLD_REPORT.md)
  label_sheet.py       emit a human-labeling sheet for definitive detection eval
  plots.py             error-bar figures (optional matplotlib)
  demo.py              self-contained worked example (no server/model)
scripts/
  run_sweep.ps1        retrieval-config × model-size sweep orchestrator
data/
  papers/*.pdf         4 open-access ML papers (system-under-test corpus)
  questions.jsonl      26 hand-curated questions (see DATASET.md)
tests/                 metric + stats + report unit tests (no downloads)
```

## Why this is a research artifact, not just a script

- **Reproduction** of a published metric, with the definitions and the
  official "necessary-member" precision rule implemented and unit-tested.
- **Reference-free faithfulness** — measures support via entailment, not
  lexical overlap, so it catches citations that are *plausible but wrong*.
- Sets up the Week 2–3 work: a controlled retrieval × model-size study with
  **bootstrap confidence intervals and significance tests**, and a novel
  lightweight NLI verifier evaluated against human labels.

See [`DATASET.md`](DATASET.md) for the data decision and its limitations.
