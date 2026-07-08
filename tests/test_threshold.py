"""Tests for the threshold sensitivity sweep (offline, KeywordNLI).

The load-bearing test is `test_analytic_sweep_matches_verify_eval`: the cached,
threshold-free scores swept at a given tau must reproduce `verify_eval.evaluate_run`
at that tau field-for-field (including the seeded bootstrap CIs). That pins the fast
analytic path to the ground-truth implementation.
"""

from __future__ import annotations

import json

import pytest

from citeval import report, threshold, verify_eval
from citeval.nli import KeywordNLI


def _p(pid, page, text):
    return {"paper_id": pid, "page": page, "text": text}


RETRIEVED = [
    _p("abc123", 2, "The Transformer relies entirely on self-attention."),
    _p("abc123", 10, "References and acknowledgements section."),
]
SELF_ATTN = "The Transformer relies entirely on self-attention"
UNSUPPORTED = "The model was trained on 4.5 million sentence pairs"


def _row(qid, answer, expected_pages):
    return {
        "id": qid,
        "question": "q?",
        "answer": answer,
        "expected_pages": expected_pages,
        "retrieved": RETRIEVED,
        "faithfulness": {},  # presence required by load_run's filter
    }


# Covers a true-positive flag+repair, a supported citation, an abstain+miss, and a
# false-positive flag on a gold page — enough variety to exercise every metric cell.
ROWS = [
    _row("q1", f"{SELF_ATTN} [abc123:10].", [2]),   # raw miss, flag tp, repair->gold
    _row("q2", f"{SELF_ATTN} [abc123:2].", [2]),    # raw hit, supported
    _row("q3", f"{UNSUPPORTED} [abc123:10].", [2]), # raw miss, flag tp, abstain
    _row("q4", f"{UNSUPPORTED} [abc123:2].", [2]),  # raw hit but flagged -> fp, abstain
]


def _write_run(runs_dir, name, rows):
    d = runs_dir / name
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(threshold, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "cfg", ROWS)
    return tmp_path


def test_analytic_sweep_matches_verify_eval(runs_dir):
    tau, n_boot, seed = 0.5, 500, 0
    # coverage=tau makes KeywordNLI.entails equivalent to entail_prob>=tau, matching
    # the real CrossEncoder where both thresholds are the same knob.
    nli = KeywordNLI(coverage=tau)

    ref = verify_eval.evaluate_run("cfg", nli, threshold=tau, n_boot=n_boot, seed=seed)
    scores = threshold.score_run("cfg", nli)
    got = threshold.metrics_at(scores, tau, n_boot=n_boot, seed=seed)

    assert got.detect_precision == pytest.approx(ref.detect_precision)
    assert got.detect_recall == pytest.approx(ref.detect_recall)
    assert got.detect_f1 == pytest.approx(ref.detect_f1)
    assert got.abstain_rate == pytest.approx(ref.abstain_rate)
    # Same per-question inputs + same seed/n_boot => identical bootstrap CIs.
    assert got.raw_gold.point == pytest.approx(ref.raw_gold.point)
    assert got.repaired_gold.point == pytest.approx(ref.repaired_gold.point)
    assert got.repaired_gold.lo == pytest.approx(ref.repaired_gold.lo)
    assert got.repaired_gold.hi == pytest.approx(ref.repaired_gold.hi)
    assert got.delta_lo == pytest.approx(ref.delta_lo)
    assert got.delta_hi == pytest.approx(ref.delta_hi)
    assert got.delta_p == pytest.approx(ref.delta_p)


def test_abstention_is_monotonic_in_tau(runs_dir):
    nli = KeywordNLI()
    scores = threshold.score_run("cfg", nli)
    taus = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    points = threshold.sweep(scores, taus, n_boot=100, seed=0)
    rates = [p.abstain_rate for p in points]
    assert rates == sorted(rates)  # raising the bar can only abstain more, never less


def test_scores_cache_roundtrip(runs_dir):
    nli = KeywordNLI()
    scores = threshold.score_run("cfg", nli)
    threshold.save_scores("cfg", scores)
    assert threshold.load_scores("cfg") == scores


def test_scores_are_threshold_independent(runs_dir):
    # Scoring never sees a threshold; the same cache serves every tau.
    nli = KeywordNLI()
    a = threshold.score_run("cfg", nli)
    b = threshold.score_run("cfg", nli)
    assert a == b
    assert all(0.0 <= s.best_prob <= 1.0 and 0.0 <= s.cited_joint_prob <= 1.0 for s in a)


def test_render_produces_markdown(runs_dir):
    nli = KeywordNLI()
    scores = threshold.score_run("cfg", nli)
    points = threshold.sweep(scores, [0.3, 0.5, 0.7], n_boot=100, seed=0)
    md = threshold.render({"cfg": points})
    assert "# Verifier threshold sensitivity" in md
    assert "`cfg`" in md
    assert "Detection F1 peaks at" in md
