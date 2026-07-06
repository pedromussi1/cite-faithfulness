# Faithfulness controlled study — results

_Bootstrap 95% CIs (10,000 resamples, seed=0). Citation precision/recall follow the ALCE (Gao et al., 2023) definitions._

## Per-configuration citation faithfulness

| Config | n | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Fully-supported |
|---|---:|---|---|---|---:|
| `dense-3b` | 26 | 0.154 [0.038, 0.308] | 0.135 [0.019, 0.269] | 0.141 [0.026, 0.282] | 12% |
| `dense-8b` | 26 | 0.173 [0.058, 0.308] | 0.183 [0.048, 0.337] | 0.173 [0.051, 0.314] | 15% |
| `hybrid+rerank-3b` | 26 | 0.250 [0.096, 0.423] | 0.244 [0.090, 0.410] | 0.246 [0.092, 0.415] | 23% |
| `hybrid+rerank-8b` | 26 | 0.308 [0.154, 0.481] | 0.301 [0.135, 0.481] | 0.297 [0.141, 0.467] | 27% |
| `hybrid-3b` | 26 | 0.192 [0.038, 0.346] | 0.192 [0.038, 0.346] | 0.192 [0.038, 0.346] | 19% |
| `hybrid-8b` | 26 | 0.308 [0.135, 0.481] | 0.301 [0.135, 0.474] | 0.304 [0.135, 0.477] | 27% |
| `rerank-3b` | 26 | 0.115 [0.000, 0.231] | 0.115 [0.000, 0.231] | 0.115 [0.000, 0.231] | 12% |
| `rerank-8b` | 26 | 0.212 [0.077, 0.385] | 0.205 [0.064, 0.372] | 0.208 [0.069, 0.377] | 19% |

## Pairwise significance vs. baseline `dense-8b`

Paired bootstrap on citation F1 (same questions), and an exact McNemar test on the per-question fully-supported outcome. `*` marks a 95% CI that excludes zero.

| Config | ΔF1 vs baseline (95% CI) | bootstrap p | McNemar p (supported) |
|---|---|---:|---:|
| `dense-3b` | -0.032 [-0.199, +0.128] | 0.7239 | 1.0000 |
| `hybrid+rerank-3b` | +0.073 [-0.104, +0.259] | 0.4436 | 0.6875 |
| `hybrid+rerank-8b` | +0.124 [+0.000, +0.272] | 0.0574 | 0.2500 |
| `hybrid-3b` | +0.019 [-0.173, +0.218] | 0.8641 | 1.0000 |
| `hybrid-8b` | +0.131 [-0.014, +0.291] | 0.0806 | 0.3750 |
| `rerank-3b` | -0.058 [-0.212, +0.090] | 0.4856 | 1.0000 |
| `rerank-8b` | +0.035 [-0.051, +0.141] | 0.4980 | 1.0000 |

## How to read this

- A CI that spans zero in the ΔF1 column means the difference from the baseline is **not** statistically distinguishable at this sample size — an honest negative result, not a win.
- Precision/recall trade off: reranking often raises precision while a wider candidate pool raises recall. F1 is the headline.
- Small n (see the `n` column) yields wide CIs; the Week-3 dataset expansion tightens them.
