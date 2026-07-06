"""Evaluate the NLI citation verifier against independent ground truth.

The key question: does a cheap local NLI verifier make citations *better* — and
can we show it without circularity? We answer with the **human-curated gold
pages** (`expected_pages`), which are independent of the NLI judge:

* **Repair lift** — for each cited sentence, does the citation land on a gold
  page? We compare the model's raw citation vs. the verifier's re-attributed
  citation (abstention counts as a miss — the fair end-to-end comparison), with
  a paired bootstrap ΔCI + p-value per configuration.

* **Detection** — the verifier flags a sentence when its own citations don't
  entail the claim. Treating "cited a non-gold page" as the (proxy) ground-truth
  for "unsupported", we report the flagger's precision / recall / F1. Swap in
  human statement labels (see `label_sheet.py`) for the definitive version.

* **Abstention** — how often the verifier declines to cite because nothing in
  the retrieved pool entails the claim (an honest "can't support this").

Runs entirely offline on archived runs (answers + stored passages), so it needs
no PaperPal and no Ollama — only the NLI model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .report import RUNS_DIR, load_run
from .stats import CI, bootstrap_ci, paired_diff
from .verifier import verify_answer


@dataclass(frozen=True)
class VerifierStats:
    name: str
    n: int
    raw_gold: CI  # sentence-level: raw citation hits a gold page
    repaired_gold: CI  # verifier's repaired citation hits a gold page
    delta_p: float  # paired bootstrap p for (repaired - raw)
    delta_lo: float
    delta_hi: float
    abstain_rate: float
    detect_precision: float
    detect_recall: float
    detect_f1: float
    n_cited_sentences: int


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_run(
    name: str, nli, *, threshold: float, n_boot: int, seed: int
) -> VerifierStats | None:
    rows = [r for r in load_run(name) if r.get("retrieved") and r.get("answer")]
    if not rows:
        return None

    raw_rate: list[float] = []  # per-question mean gold-hit (raw)
    rep_rate: list[float] = []  # per-question mean gold-hit (repaired)
    abstains: list[float] = []
    tp = fp = fn = 0  # detection vs gold-page proxy
    n_cited = 0

    for row in rows:
        gold = set(row.get("expected_pages", []))
        verdicts = verify_answer(row["answer"], row["retrieved"], nli, threshold=threshold)
        if not verdicts:
            continue
        raw_hits, rep_hits, abst = [], [], []
        for v in verdicts:
            n_cited += 1
            raw_hit = 1 if (v.original_pages & gold) else 0
            rep_hit = 1 if (v.repaired is not None and v.repaired[1] in gold) else 0
            raw_hits.append(raw_hit)
            rep_hits.append(rep_hit)
            abst.append(1 if v.abstained else 0)
            # Detection: flagged-unsupported = not original_supported; proxy
            # truth for "unsupported" = the raw citation missed every gold page.
            flagged = not v.original_supported
            truth_unsupported = raw_hit == 0
            if flagged and truth_unsupported:
                tp += 1
            elif flagged and not truth_unsupported:
                fp += 1
            elif not flagged and truth_unsupported:
                fn += 1
        raw_rate.append(sum(raw_hits) / len(raw_hits))
        rep_rate.append(sum(rep_hits) / len(rep_hits))
        abstains.append(sum(abst) / len(abst))

    if not raw_rate:
        return None
    diff = paired_diff(rep_rate, raw_rate, n_boot=n_boot, seed=seed)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return VerifierStats(
        name=name,
        n=len(raw_rate),
        raw_gold=bootstrap_ci(raw_rate, n_boot=n_boot, seed=seed),
        repaired_gold=bootstrap_ci(rep_rate, n_boot=n_boot, seed=seed),
        delta_p=diff.p_value,
        delta_lo=diff.lo,
        delta_hi=diff.hi,
        abstain_rate=sum(abstains) / len(abstains),
        detect_precision=prec,
        detect_recall=rec,
        detect_f1=_f1(prec, rec),
        n_cited_sentences=n_cited,
    )


def render(stats: list[VerifierStats], *, threshold: float) -> str:
    lines = [
        "# NLI citation verifier — evaluation",
        "",
        f"_Verifier threshold={threshold}. Ground truth = human-curated gold pages "
        "(`expected_pages`), independent of the NLI judge — so this is not circular._",
        "",
        "## Repair: does re-attribution land on a gold page more often?",
        "",
        "Sentence-level gold-hit rate (abstention counts as a miss), per config, "
        "with bootstrap 95% CIs. ΔCI/​p from a paired bootstrap (repaired − raw). "
        "`*` = 95% CI excludes zero.",
        "",
        "| Config | raw → gold (95% CI) | repaired → gold (95% CI) | Δ (95% CI) | p | abstain |",
        "|---|---|---|---|---:|---:|",
    ]
    for s in stats:
        star = " *" if (s.delta_lo > 0 or s.delta_hi < 0) else ""
        lines.append(
            f"| `{s.name}` "
            f"| {s.raw_gold.point:.3f} [{s.raw_gold.lo:.3f}, {s.raw_gold.hi:.3f}] "
            f"| {s.repaired_gold.point:.3f} [{s.repaired_gold.lo:.3f}, {s.repaired_gold.hi:.3f}] "
            f"| {s.repaired_gold.point - s.raw_gold.point:+.3f} "
            f"[{s.delta_lo:+.3f}, {s.delta_hi:+.3f}]{star} "
            f"| {s.delta_p:.4f} | {s.abstain_rate:.0%} |"
        )
    lines += [
        "",
        "## Detection (vs. gold-page proxy)",
        "",
        "Flagger = a sentence whose own citations don't entail it. Proxy truth = "
        "the citation missed every gold page. (Human statement labels give the "
        "definitive number — see `label_sheet.py`.)",
        "",
        "| Config | precision | recall | F1 | cited sents |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| `{s.name}` | {s.detect_precision:.3f} | {s.detect_recall:.3f} "
            f"| {s.detect_f1:.3f} | {s.n_cited_sentences} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Repair lift** is the headline: a positive, significant Δ means the "
        "cheap NLI verifier re-attributes citations to the correct (gold) page "
        "more often than the raw model — a genuine, non-circular improvement.",
        "- **Abstain** is the safety valve: when no retrieved passage entails the "
        "claim, the verifier declines rather than assert a citation. High abstain "
        "with low raw gold-hit points at a *retrieval* ceiling, not a verifier fault.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import contextlib
    import sys

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=str(RUNS_DIR / "VERIFIER_REPORT.md"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from .nli import get_nli

    names = args.runs
    if args.all:
        names = sorted(p.parent.name for p in RUNS_DIR.glob("*/results.jsonl"))
    if not names:
        print("Specify --runs <names...> or --all.")
        return 2

    nli = get_nli(args.nli, **({} if args.nli == "mock" else {"threshold": args.threshold}))
    stats: list[VerifierStats] = []
    for name in names:
        print(f"verifying {name}...", flush=True)
        s = evaluate_run(name, nli, threshold=args.threshold, n_boot=args.n_boot, seed=args.seed)
        if s:
            stats.append(s)

    report = render(stats, threshold=args.threshold)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
