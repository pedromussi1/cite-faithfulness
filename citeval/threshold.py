"""Verifier threshold sensitivity analysis — the abstention/accuracy trade-off.

The NLI verifier (`verifier.py`) has a single knob, the entailment threshold
``tau``. It gates two decisions:

* **flag** — a cited sentence is judged *supported* iff its own cited passage
  entails the claim with probability >= ``tau``; otherwise the verifier flags it.
* **abstain** — repair declines (abstains) iff *no* retrieved passage entails
  the claim with probability >= ``tau``.

Crucially, both decisions are just thresholds over **continuous NLI scores that
do not depend on ``tau``**: the entailment probability of the model's own cited
passage, and the best entailment probability over the retrieved pool. So we run
the NLI model **once** per run to cache those two scores per cited sentence, then
sweep ``tau`` purely analytically — instant, reproducible, and torch-free. At any
single ``tau`` this reproduces ``verify_eval`` exactly (see the tests).

The output curve answers the practical question: *where should you set the
threshold?* Raising ``tau`` makes the flagger more aggressive (higher recall on
unsupported citations) but makes repair abstain more often (fewer re-attributed
citations actually land). This module quantifies that trade-off per config.

Usage:

    # score once with the real NLI (caches runs/<cfg>/verifier_scores.json), then sweep
    python -m citeval.threshold --all --nli cross-encoder/nli-deberta-v3-base

    # re-sweep from cache with no model download (torch-free)
    python -m citeval.threshold --all --from-cache
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import (
    build_passage_map,
    parse_citations,
    split_sentences,
    strip_citations,
)
from .report import RUNS_DIR, load_run
from .stats import CI, bootstrap_ci, mean, paired_diff

SCORES_FILENAME = "verifier_scores.json"
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class SentenceScore:
    """Threshold-independent NLI scores for one cited sentence.

    Everything the threshold sweep needs is derivable from these six fields, so
    the (expensive) NLI pass runs once and every ``tau`` is free arithmetic.
    """

    qid: str
    raw_hit: int  # 1 if the model's own citation pages intersect the gold pages
    has_cited: bool  # the model cited at least one *retrieved* passage
    cited_joint_prob: float  # entail_prob(model's cited passage, claim) -> flag
    best_prob: float  # max entail_prob over the retrieved pool -> abstain
    best_in_gold: int  # 1 if the best-entailing pool passage sits on a gold page


@dataclass(frozen=True)
class ThresholdPoint:
    tau: float
    abstain_rate: float
    detect_precision: float
    detect_recall: float
    detect_f1: float
    raw_gold: CI
    repaired_gold: CI
    delta_lo: float
    delta_hi: float
    delta_p: float


def score_run(name: str, nli) -> list[SentenceScore]:
    """Run the NLI model once over a run's cited sentences, caching the two
    continuous scores that the threshold later gates. Mirrors the scoring inside
    ``verifier.verify_answer`` / ``verify_eval.evaluate_run`` exactly, but stores
    probabilities instead of applying a threshold."""
    scores: list[SentenceScore] = []
    for row in load_run(name):
        if not (row.get("retrieved") and row.get("answer")):
            continue
        gold = set(row.get("expected_pages", []))
        passages = build_passage_map(row["retrieved"])
        pool = list(passages.items())  # [((paper_id, page), text), ...]

        for sent in split_sentences(row["answer"]):
            keys = parse_citations(sent)
            if not keys:
                continue
            hyp = strip_citations(sent)
            if not hyp:
                continue

            cited_joint = "\n\n".join(t for k in keys if (t := passages.get(k, "")))
            has_cited = bool(cited_joint)
            cited_joint_prob = nli.entail_prob(cited_joint, hyp) if has_cited else 0.0

            best_prob, best_page = 0.0, None
            if pool:
                probs = nli.entail_probs([(text, hyp) for _, text in pool])
                best_i = max(range(len(probs)), key=lambda i: probs[i])
                best_prob, best_page = probs[best_i], pool[best_i][0][1]

            original_pages = {page for _, page in keys}
            scores.append(
                SentenceScore(
                    qid=str(row["id"]),
                    raw_hit=1 if (original_pages & gold) else 0,
                    has_cited=has_cited,
                    cited_joint_prob=float(cited_joint_prob),
                    best_prob=float(best_prob),
                    best_in_gold=1 if (best_page in gold) else 0,
                )
            )
    return scores


def save_scores(name: str, scores: list[SentenceScore]) -> Path:
    out = RUNS_DIR / name / SCORES_FILENAME
    out.write_text(json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8")
    return out


def load_scores(name: str) -> list[SentenceScore]:
    p = RUNS_DIR / name / SCORES_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"No cached scores for {name} (looked at {p}). "
                                "Run without --from-cache first.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [SentenceScore(**d) for d in raw]


def metrics_at(
    scores: list[SentenceScore], tau: float, *, n_boot: int, seed: int
) -> ThresholdPoint:
    """Compute the verifier metrics at threshold ``tau`` from cached scores.

    Aggregation matches ``verify_eval.evaluate_run`` field-for-field (per-question
    means, then a bootstrap CI; detection counted over all cited sentences), so at
    a given ``tau`` this returns the same numbers — just without touching the NLI.
    """
    by_q: dict[str, list[SentenceScore]] = defaultdict(list)
    for s in scores:
        by_q[s.qid].append(s)

    raw_rate: list[float] = []
    rep_rate: list[float] = []
    abst_rate: list[float] = []
    tp = fp = fn = 0

    for sents in by_q.values():
        rh, ph, ab = [], [], []
        for s in sents:
            original_supported = s.has_cited and s.cited_joint_prob >= tau
            abstained = s.best_prob < tau
            rep_hit = 0 if abstained else s.best_in_gold
            rh.append(s.raw_hit)
            ph.append(rep_hit)
            ab.append(1 if abstained else 0)

            flagged = not original_supported
            truth_unsupported = s.raw_hit == 0
            if flagged and truth_unsupported:
                tp += 1
            elif flagged and not truth_unsupported:
                fp += 1
            elif not flagged and truth_unsupported:
                fn += 1
        raw_rate.append(mean(rh))
        rep_rate.append(mean(ph))
        abst_rate.append(mean(ab))

    diff = paired_diff(rep_rate, raw_rate, n_boot=n_boot, seed=seed)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return ThresholdPoint(
        tau=tau,
        abstain_rate=mean(abst_rate) if abst_rate else 0.0,
        detect_precision=prec,
        detect_recall=rec,
        detect_f1=f1,
        raw_gold=bootstrap_ci(raw_rate, n_boot=n_boot, seed=seed),
        repaired_gold=bootstrap_ci(rep_rate, n_boot=n_boot, seed=seed),
        delta_lo=diff.lo,
        delta_hi=diff.hi,
        delta_p=diff.p_value,
    )


def sweep(
    scores: list[SentenceScore], thresholds, *, n_boot: int, seed: int
) -> list[ThresholdPoint]:
    return [metrics_at(scores, t, n_boot=n_boot, seed=seed) for t in thresholds]


def _best_f1(points: list[ThresholdPoint]) -> ThresholdPoint:
    return max(points, key=lambda p: (p.detect_f1, -p.tau))


def render(sweeps: dict[str, list[ThresholdPoint]]) -> str:
    lines = [
        "# Verifier threshold sensitivity",
        "",
        "How the verifier's one knob — the entailment threshold `tau` — trades off "
        "**detection** (flagging unsupported citations) against **abstention** "
        "(declining to re-attribute when nothing entails). Ground truth = "
        "human-curated gold pages, independent of the NLI judge. Scores are cached "
        "per run, so this whole table is an analytic sweep over a single NLI pass.",
        "",
        "For each config: detection precision/recall/F1 (vs the gold-page proxy for "
        "\"unsupported\"), the repair gold-hit rate with its Δ vs the raw model "
        "(paired bootstrap; `*` = 95% CI excludes 0), and the abstention rate.",
        "",
    ]
    for name, points in sweeps.items():
        star_tau = _best_f1(points)
        lines += [
            f"## `{name}`",
            "",
            f"Detection F1 peaks at **tau={star_tau.tau:.2f}** "
            f"(P={star_tau.detect_precision:.3f}, R={star_tau.detect_recall:.3f}, "
            f"F1={star_tau.detect_f1:.3f}, abstain={star_tau.abstain_rate:.0%}).",
            "",
            "| tau | det P | det R | det F1 | repaired→gold | Δ vs raw (95% CI) | abstain |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in points:
            excl0 = p.delta_lo > 0 or p.delta_hi < 0
            star = " *" if excl0 else ""
            mark = " **" if p.tau == star_tau.tau else ""
            end = "**" if p.tau == star_tau.tau else ""
            lines.append(
                f"|{mark} {p.tau:.2f}{end} | {p.detect_precision:.3f} | "
                f"{p.detect_recall:.3f} | {p.detect_f1:.3f} | "
                f"{p.repaired_gold.point:.3f} | "
                f"{p.repaired_gold.point - p.raw_gold.point:+.3f} "
                f"[{p.delta_lo:+.3f}, {p.delta_hi:+.3f}]{star} | {p.abstain_rate:.0%} |"
            )
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "- **The trade-off:** raising `tau` flags more aggressively (detection recall "
        "up) but pushes repair to abstain more (fewer re-attributions land), so the "
        "repaired gold-hit rate falls. The detection-F1 peak is a principled default.",
        "- **Abstention is diagnostic, not failure:** high abstain with low raw "
        "gold-hit means the *retrieved pool* rarely contains an entailing passage — "
        "a retrieval ceiling the verifier can't fix, only refuse to paper over.",
        "- A Δ whose 95% CI still spans 0 at every `tau` (as at n=26) is an honest "
        "non-result on repair lift; the flagger can still be useful on its own.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import contextlib
    import sys

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated tau grid.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Use cached verifier_scores.json (no NLI model load).",
    )
    parser.add_argument("--out", default=str(RUNS_DIR / "THRESHOLD_REPORT.md"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = args.runs
    if args.all:
        names = sorted(p.parent.name for p in RUNS_DIR.glob("*/results.jsonl"))
    if not names:
        print("Specify --runs <names...> or --all.")
        return 2

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    nli = None
    if not args.from_cache:
        from .nli import get_nli

        nli = get_nli(args.nli)

    sweeps: dict[str, list[ThresholdPoint]] = {}
    for name in names:
        if args.from_cache:
            scores = load_scores(name)
        else:
            print(f"scoring {name}...", flush=True)
            scores = score_run(name, nli)
            save_scores(name, scores)
        sweeps[name] = sweep(scores, thresholds, n_boot=args.n_boot, seed=args.seed)

    report = render(sweeps)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
