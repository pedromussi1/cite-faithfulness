# NLI citation verifier — evaluation

_Verifier threshold=0.5. Ground truth = human-curated gold pages (`expected_pages`), independent of the NLI judge — so this is not circular._

## Repair: does re-attribution land on a gold page more often?

Sentence-level gold-hit rate (abstention counts as a miss), per config, with bootstrap 95% CIs. ΔCI/​p from a paired bootstrap (repaired − raw). `*` = 95% CI excludes zero.

| Config | raw → gold (95% CI) | repaired → gold (95% CI) | Δ (95% CI) | p | abstain |
|---|---|---|---|---:|---:|
| `dense-3b` | 0.438 [0.219, 0.656] | 0.156 [0.000, 0.344] | -0.281 [-0.562, +0.000] | 0.0694 | 59% |
| `dense-8b` | 0.477 [0.273, 0.682] | 0.250 [0.091, 0.432] | -0.227 [-0.477, +0.023] | 0.0982 | 48% |
| `hybrid+rerank-3b` | 0.658 [0.447, 0.842] | 0.447 [0.237, 0.658] | -0.211 [-0.474, +0.053] | 0.1806 | 45% |
| `hybrid+rerank-8b` | 0.548 [0.333, 0.762] | 0.357 [0.167, 0.571] | -0.190 [-0.429, +0.024] | 0.1214 | 45% |
| `hybrid-3b` | 0.450 [0.250, 0.650] | 0.450 [0.250, 0.650] | +0.000 [-0.300, +0.300] | 1.0000 | 45% |
| `hybrid-8b` | 0.500 [0.304, 0.696] | 0.391 [0.196, 0.587] | -0.109 [-0.304, +0.087] | 0.3346 | 48% |
| `rerank-3b` | 0.361 [0.167, 0.583] | 0.250 [0.056, 0.444] | -0.111 [-0.417, +0.222] | 0.5439 | 58% |
| `rerank-8b` | 0.521 [0.333, 0.708] | 0.271 [0.104, 0.458] | -0.250 [-0.500, +0.000] | 0.0594 | 56% |

## Detection (vs. gold-page proxy)

Flagger = a sentence whose own citations don't entail it. Proxy truth = the citation missed every gold page. (Human statement labels give the definitive number — see `label_sheet.py`.)

| Config | precision | recall | F1 | cited sents |
|---|---:|---:|---:|---:|
| `dense-3b` | 0.625 | 0.833 | 0.714 | 20 |
| `dense-8b` | 0.550 | 0.786 | 0.647 | 26 |
| `hybrid+rerank-3b` | 0.462 | 0.857 | 0.600 | 20 |
| `hybrid+rerank-8b` | 0.600 | 0.750 | 0.667 | 24 |
| `hybrid-3b` | 0.684 | 0.929 | 0.788 | 24 |
| `hybrid-8b` | 0.714 | 0.882 | 0.789 | 30 |
| `rerank-3b` | 0.722 | 0.929 | 0.813 | 21 |
| `rerank-8b` | 0.565 | 0.929 | 0.703 | 29 |

## How to read this

- **Repair lift** is the headline: a positive, significant Δ means the cheap NLI verifier re-attributes citations to the correct (gold) page more often than the raw model — a genuine, non-circular improvement.
- **Abstain** is the safety valve: when no retrieved passage entails the claim, the verifier declines rather than assert a citation. High abstain with low raw gold-hit points at a *retrieval* ceiling, not a verifier fault.
