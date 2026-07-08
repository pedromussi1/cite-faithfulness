# Are RAG Citations Faithful? A Controlled Study of Citation Faithfulness in a Local Retrieval-Augmented QA System

**Author:** Pedro Mussi
**Status:** Working paper (v1.0), part of the `cite-faithfulness` project.
**Code & data:** this repository. All experiments are local, offline-reproducible, and $0 (no paid APIs).

---

## Abstract

Retrieval-augmented generation (RAG) systems increasingly answer questions with inline citations
of the form `[document:page]`. A common way to "evaluate" such citations is to check whether the
cited *page number* matches a gold page — a lexical proxy that says nothing about whether the cited
passage actually **supports** the claim. We reproduce the citation precision/recall metric of ALCE
(Gao et al., 2023) — which credits a citation only when a Natural Language Inference (NLI) model
judges the cited passage to *entail* the sentence — and apply it as an external evaluator to a local
RAG QA system (PaperPal) over four ML papers and 26 questions. Three findings emerge. **(1)**
Faithfulness is low across every configuration (best mean citation F1 ≈ 0.30; at most 27% of answers
fully supported), and it diverges sharply from the page-overlap proxy — the gap that motivates the
study. **(2)** Retrieval quality, not model size, is the dominant lever: hybrid dense+sparse
retrieval roughly doubles faithfulness over dense-only for the larger model (F1 0.17 → 0.30), though
at n=26 no single pairwise difference is statistically significant (all Δ CIs cross zero). **(3)** A
cheap local NLI *verifier* reliably **detects** unsupported citations (flag F1 0.60–0.81), but naive
**auto-repair** (re-attributing to the best-entailing retrieved passage) does not improve gold-page
accuracy, because the verifier must abstain 45–59% of the time — the entailing passage is simply not
in the retrieved pool. This implicates *retrieval* as the bottleneck and argues for using such a
verifier to **flag, not silently rewrite**. We release the metric, the controlled-study harness, the
verifier, a threshold-sensitivity analysis, and all runs.

---

## 1. Introduction

RAG systems are widely deployed to make LLM answers checkable: each claim comes with a citation the
user can click. But a citation is only useful if the cited passage *actually supports* the claim.
Many pipelines (including the system under test here) self-report citation quality by checking
whether the cited page number appears among the gold pages for a question. This **page-overlap
proxy** is attractive because it needs no model, but it conflates two very different things: *did the
model point at the right page*, and *does the passage on that page entail the sentence*. A model can
cite the correct page for the wrong reason, or cite a plausible-looking page whose text does not
support the claim at all.

This project asks a narrow, measurable question: **when a local RAG system cites its sources, how
often are those citations actually faithful — and does the answer differ from what the page-overlap
proxy reports?** We answer it by reproducing the ALCE citation metric (Gao et al., 2023) as an
*independent* evaluator that runs against the RAG system over its HTTP API, with no coupling to the
system's internals.

Contributions:

1. A faithful, tested reproduction of the ALCE citation precision/recall metric (including the
   official "necessary-member" rule for multi-citation sentences and hallucinated-citation
   handling), driven against a live RAG system over a streaming API.
2. A controlled study across 4 retrieval configurations × 2 model sizes with proper uncertainty
   quantification (bootstrap confidence intervals, paired bootstrap ΔF1, exact McNemar), reported
   honestly including the negative/underpowered results.
3. A novel, cheap **NLI citation verifier** that flags and optionally repairs unsupported citations,
   with a *non-circular* evaluation against human-curated gold pages (never against its own judge),
   and a threshold-sensitivity analysis of the flag/abstain trade-off.

## 2. Related work

**ALCE** (Gao et al., EMNLP 2023) introduced automatic citation precision and recall for LLM
answers, using an NLI model to decide whether cited passages entail a statement, and the
"necessary-member" rule that penalizes citations that are individually unnecessary. We reproduce
this metric with a small, local NLI checkpoint rather than the large TRUE/T5-11B model ALCE used, to
keep the study on a single consumer GPU at $0.

**RAGAS** (Es et al., 2023) proposes reference-free RAG metrics including *faithfulness* (are answer
claims supported by retrieved context) and *answer/context relevance*, typically judged by a strong
LLM. Our focus is narrower and citation-level: not "is the answer supported somewhere in context"
but "does *this cited passage* support *this sentence*."

**FactScore** (Min et al., EMNLP 2023) decomposes long-form generations into atomic facts and scores
each against a knowledge source. We share the atomic, per-statement spirit but operate on the model's
*own emitted citations* as the unit of analysis, which is what a user actually clicks.

Relative to these, this study's angle is (a) an explicit, measured contrast between a page-overlap
proxy and true entailment-based faithfulness on the *same* system, and (b) a verifier that acts on
citations at inference time, evaluated without circularity.

## 3. Method

**System under test (SUT).** PaperPal, a local RAG QA system, is treated as a black box. The
evaluator drives it over its `/query` Server-Sent-Events API: a `retrieved` event carries the
candidate passages (with `paper_id`, `page`, and text), and a `done` event carries the answer with
inline `[paper_id:page]` citations. There is no import coupling — the evaluator would work against
any system that speaks the same API.

**Citation metric (ALCE reproduction).** For each answer we split sentences, parse their citations,
and for each cited sentence ask an NLI model whether the cited passage(s) entail the
citation-stripped sentence. *Citation recall* credits a sentence whose citations jointly entail it;
*citation precision* additionally applies the necessary-member rule (a citation is imprecise if the
sentence is still entailed without it). Hallucinated citations (to passages never retrieved) are
detected and penalized. We report per-configuration precision, recall, F1, and the fraction of
answers that are *fully supported*.

**NLI judge.** A local `cross-encoder/nli-deberta-v3-base` cross-encoder provides entailment
probabilities. Because small cross-encoders truncate at 512 tokens — silently cutting support that
sits near the end of a long retrieved passage — we score each hypothesis against **overlapping
sentence windows** of the passage and take the maximum entailment. This windowing fix was material:
on a fact buried at the end of a wide passage, entailment rose from 0.006 to 0.997. The NLI backend
is pluggable; a deterministic keyword stub is used for unit tests and CI so the metric logic is
pinned without downloading any model.

**Controlled study.** We sweep 4 retrieval configurations — `dense`, `rerank`, `hybrid`,
`hybrid+rerank` — crossed with 2 generator sizes (`llama3.2:3b`, `llama3.1:8b`), restarting the SUT
per cell so configurations do not leak. Retrieved passages are archived so every downstream metric
(and the verifier) can be recomputed offline without re-querying the models.

**NLI citation verifier (novel contribution).** Given an answer and its retrieved pool, the verifier
(i) **flags** each cited sentence whose own citations do not entail it, and (ii) attempts **repair**
by re-attributing the citation to the retrieved passage that best entails the claim, *abstaining* if
nothing in the pool clears an entailment threshold `tau`. Because repair uses NLI, scoring the
repaired answer with the same NLI would be circular; we therefore evaluate the verifier only against
**independent human-curated gold pages** (and provide a human-statement-label sheet generator for the
definitive detection metric). A threshold-sensitivity analysis (§5.4) characterizes the
detection-vs-abstention trade-off as `tau` varies; the scores are cached once so the sweep is exact
and reproducible.

**Statistics.** All point estimates carry percentile bootstrap 95% CIs (10,000 resamples, seeded).
Configuration comparisons use a paired bootstrap on per-question F1 (ΔF1 with a two-sided p) and an
exact McNemar test on the per-question fully-supported outcome. We treat a Δ whose 95% CI includes
zero as a non-result, and say so.

## 4. Experimental setup

- **Corpus:** four ML papers — *Attention Is All You Need*, *BERT*, *LoRA*, *CLIP* — identified by
  content hash so the evaluator and SUT provably index the same PDFs.
- **Questions:** 26 hand-curated questions (factoid / explanation / comparison categories), each with
  a gold answer and human-curated `expected_pages`.
- **Models:** `llama3.1:8b` and `llama3.2:3b` via Ollama (local, $0).
- **NLI:** `cross-encoder/nli-deberta-v3-base` (local, fits 12 GB).
- **Reproducibility:** CI runs the full metric/verifier/stats test suite with the keyword-NLI stub
  (no torch download); real numbers come from the archived runs in `runs/`.

## 5. Results

### 5.1 Faithfulness is low, and diverges from the page proxy

Across all eight configurations, mean citation F1 ranges from **0.115** (`rerank-3b`) to **0.304**
(`hybrid-8b`), and the fraction of *fully supported* answers never exceeds **27%** (Table 1). The
system's own page-overlap proxy is far more optimistic — a majority of answers cite a gold page
(§5.3, "raw → gold" columns reach 0.44–0.66) even though only a minority are entailment-supported.
That divergence between "cited the right page" and "the passage supports the claim" is precisely the
gap the page proxy hides, and the central motivation for entailment-based evaluation.

**Table 1 — Per-configuration citation faithfulness (95% CI, n=26).**

| Config | Precision | Recall | F1 | Fully-supported |
|---|---|---|---|---:|
| `dense-3b` | 0.154 [0.038, 0.308] | 0.135 [0.019, 0.269] | 0.141 [0.026, 0.282] | 12% |
| `rerank-3b` | 0.115 [0.000, 0.231] | 0.115 [0.000, 0.231] | 0.115 [0.000, 0.231] | 12% |
| `hybrid-3b` | 0.192 [0.038, 0.346] | 0.192 [0.038, 0.346] | 0.192 [0.038, 0.346] | 19% |
| `hybrid+rerank-3b` | 0.250 [0.096, 0.423] | 0.244 [0.090, 0.410] | 0.246 [0.092, 0.415] | 23% |
| `dense-8b` | 0.173 [0.058, 0.308] | 0.183 [0.048, 0.337] | 0.173 [0.051, 0.314] | 15% |
| `rerank-8b` | 0.212 [0.077, 0.385] | 0.205 [0.064, 0.372] | 0.208 [0.069, 0.377] | 19% |
| `hybrid-8b` | 0.308 [0.135, 0.481] | 0.301 [0.135, 0.474] | 0.304 [0.135, 0.477] | 27% |
| `hybrid+rerank-8b` | 0.308 [0.154, 0.481] | 0.301 [0.135, 0.481] | 0.297 [0.141, 0.467] | 27% |

### 5.2 Retrieval is the dominant lever — but n=26 is underpowered

The ordering is consistent: hybrid retrieval helps most (for the 8B model, dense F1 0.173 → hybrid
0.304), the 8B generator beats 3B within each retrieval setting, and reranking alone is weak. But
against the `dense-8b` baseline, **no** pairwise ΔF1 reaches significance — every 95% CI includes
zero (Table 2). The closest is `hybrid+rerank-8b` at ΔF1 +0.124 [0.000, 0.272], p=0.057. This is an
honest underpowered result: the effect direction is stable and plausibly real, but n=26 cannot
confirm it. It directly motivates the dataset-expansion work (§6).

**Table 2 — Pairwise ΔF1 vs. `dense-8b` (paired bootstrap; McNemar on fully-supported).**

| Config | ΔF1 vs baseline (95% CI) | bootstrap p | McNemar p |
|---|---|---:|---:|
| `hybrid-8b` | +0.131 [-0.014, +0.291] | 0.081 | 0.375 |
| `hybrid+rerank-8b` | +0.124 [+0.000, +0.272] | 0.057 | 0.250 |
| `rerank-8b` | +0.035 [-0.051, +0.141] | 0.498 | 1.000 |
| `hybrid+rerank-3b` | +0.073 [-0.104, +0.259] | 0.444 | 0.688 |
| `hybrid-3b` | +0.019 [-0.173, +0.218] | 0.864 | 1.000 |
| `dense-3b` | -0.032 [-0.199, +0.128] | 0.724 | 1.000 |
| `rerank-3b` | -0.058 [-0.212, +0.090] | 0.486 | 1.000 |

### 5.3 The verifier detects unfaithful citations but should not auto-repair

Evaluated against independent gold pages (Table 3), the NLI verifier is a **good detector**: it flags
unsupported citations with F1 between **0.60 and 0.81** and recall 0.75–0.93 — it rarely misses a bad
citation. But naive **auto-repair hurts**: re-attributing to the best-entailing retrieved passage
*lowers* the gold-hit rate in 7 of 8 configurations (none significant), because the verifier abstains
**45–59%** of the time. Abstention is the honest signal here: when no retrieved passage entails the
claim, the entailing evidence was never retrieved, so there is nothing correct to repair *to*. The
takeaway is a design rule — **use the verifier to flag, not to silently rewrite** — and a diagnosis:
the ceiling is **retrieval**, not generation or the judge.

**Table 3 — Verifier at tau=0.5 (gold-page ground truth, n=26).**

| Config | detect P | detect R | detect F1 | raw → gold | repaired → gold | abstain |
|---|---:|---:|---:|---|---|---:|
| `dense-3b` | 0.625 | 0.833 | 0.714 | 0.438 | 0.156 | 59% |
| `dense-8b` | 0.550 | 0.786 | 0.647 | 0.477 | 0.250 | 48% |
| `hybrid-3b` | 0.684 | 0.929 | 0.788 | 0.450 | 0.450 | 45% |
| `hybrid-8b` | 0.714 | 0.882 | 0.789 | 0.500 | 0.391 | 48% |
| `hybrid+rerank-3b` | 0.462 | 0.857 | 0.600 | 0.658 | 0.447 | 45% |
| `hybrid+rerank-8b` | 0.600 | 0.750 | 0.667 | 0.548 | 0.357 | 45% |
| `rerank-3b` | 0.722 | 0.929 | 0.813 | 0.361 | 0.250 | 58% |
| `rerank-8b` | 0.565 | 0.929 | 0.703 | 0.521 | 0.271 | 56% |

### 5.4 Threshold sensitivity: the detection/abstention trade-off

The verifier's single knob is the entailment threshold `tau`. Because it gates only decisions over
continuous NLI scores that do not themselves depend on `tau`, we cache those scores once per run and
sweep `tau ∈ {0.1, …, 0.9}` analytically (`runs/THRESHOLD_REPORT.md`). Two robust patterns emerge.

**Detection barely moves with `tau`.** Within every configuration, detection F1 varies by less than
~0.06 across the entire threshold range (e.g. `dense-8b` 0.647–0.688; `rerank-3b` 0.813–0.839;
`hybrid-8b` 0.732–0.789). The flag decision is effectively bimodal — a cited passage either clearly
entails the claim or clearly does not — so moving the threshold rarely flips a flag. This is a useful
practical property: **the flagger works without careful tuning.** Detection F1 peaks at a *low*
threshold (`tau=0.1`–`0.2`) in seven of eight configs, where abstention is also lowest (32–53%).

**Repair only gets worse as `tau` rises.** Abstention increases monotonically with `tau` (e.g.
`dense-8b` 32% → 73%; `rerank-8b` 44% → 77%), so the repaired gold-hit rate falls and the paired Δ
against the raw model becomes *more* negative. The repair Δ is **never significantly positive at any
threshold in any configuration**, and at high `tau` (≥0.7) it turns significantly *negative* (95% CI
excludes 0) in five of eight configs — aggressive abstention actively destroys correct citations the
model already had. There is simply no operating point at which naive auto-repair becomes a win on
this data.

Together these reinforce §5.3's design rule: run the verifier at a **low threshold as a detector**
(good, tuning-insensitive F1 at modest abstention) and surface flags to the user, rather than raising
the threshold in pursuit of a repair lift that never materializes. The limiting factor remains
retrieval — the entailing passage is often absent from the pool, so no threshold can recover it.

**Table 4 — Threshold extremes for the 8B configs (detection F1 / abstention; repair Δ vs raw).**

| Config | `tau`=0.1 | `tau`=0.5 | `tau`=0.9 |
|---|---|---|---|
| `dense-8b` | F1 0.688 / 32% / Δ−0.068 | F1 0.647 / 48% / Δ−0.227 | F1 0.686 / 73% / Δ−0.318 \* |
| `rerank-8b` | F1 0.722 / 44% / Δ−0.146 | F1 0.703 / 56% / Δ−0.250 | F1 0.684 / 77% / Δ−0.333 \* |
| `hybrid-8b` | F1 0.757 / 41% / Δ−0.065 | F1 0.789 / 48% / Δ−0.109 | F1 0.732 / 72% / Δ−0.261 \* |
| `hybrid+rerank-8b` | F1 0.615 / 43% / Δ−0.190 | F1 0.667 / 45% / Δ−0.190 | F1 0.690 / 67% / Δ−0.286 \* |

\* Δ 95% CI excludes zero (a significant *decrease* in gold-hit vs. the raw model). Full grid in
`runs/THRESHOLD_REPORT.md`.

## 6. Limitations

- **Sample size.** n=26 questions yields wide CIs; the retrieval effect (§5.2) is directionally
  clear but not significant. Expanding to ~100–200 questions is the single highest-value next step.
- **Gold-page proxy.** The verifier's detection metric uses "cited a non-gold page" as a proxy for
  "unsupported." This is independent of the NLI judge (so not circular), but it is coarser than
  human per-statement labels; a label sheet generator is included to produce the definitive metric.
- **Single NLI checkpoint.** Faithfulness scores are relative to one small DeBERTa-MNLI model; a
  larger or ensemble NLI judge (as in ALCE) could shift absolute numbers. The checkpoint is a pinned,
  auditable config knob, and the windowing fix mitigates the most severe failure mode (truncation).
- **Domain.** Four ML papers and hand-curated questions; generalization to other corpora is untested.
- **Local models.** Results are for two Ollama models; larger hosted models may behave differently.

## 7. Conclusion

Checking that a RAG citation points at the right *page* substantially overstates how often the cited
*passage* actually supports the claim. Measured with a reproduced ALCE entailment metric, a local RAG
system is faithful on at most ~30% of answers, and improving faithfulness is mostly about improving
**retrieval**. A cheap local NLI verifier is a dependable *detector* of unfaithful citations but a
poor *auto-repairer* — its frequent, honest abstentions localize the bottleneck to retrieval rather
than generation. All components are released as a small, tested, offline-reproducible toolkit.

## References

- Gao, T., Yen, H., Yu, J., Chen, D. (2023). *Enabling Large Language Models to Generate Text with
  Citations.* EMNLP 2023. (ALCE)
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2023). *RAGAS: Automated Evaluation of
  Retrieval Augmented Generation.*
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P., Iyyer, M., Zettlemoyer, L., Hajishirzi,
  H. (2023). *FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text
  Generation.* EMNLP 2023.
