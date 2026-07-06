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

Status: **Week 1** — metric reproduced and unit-tested; offline driver wired to
PaperPal's `/query` API; Week-1 dataset pinned. Controlled study (retrieval
configs × model sizes, bootstrap CIs) and a novel NLI verifier follow in
Weeks 2–3. See [`PLAN` in the parent repo](../) for the full arc.

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

## Layout

```
citeval/
  metrics.py           ALCE citation precision/recall/F1 (the reproduction)
  nli.py               NLI backends: CrossEncoderNLI (real) + KeywordNLI (tests)
  client.py            async PaperPal /query SSE client
  run_faithfulness.py  offline eval driver → runs/<name>/
  demo.py              self-contained worked example (no server/model)
data/
  papers/*.pdf         4 open-access ML papers (system-under-test corpus)
  questions.jsonl      26 hand-curated questions (see DATASET.md)
tests/                 metric unit tests (KeywordNLI, no downloads)
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
