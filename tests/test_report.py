"""Report generation over synthetic runs — no server or NLI model needed."""

from __future__ import annotations

import json
from pathlib import Path

from citeval.plots import figure_series
from citeval.report import load_run, render_report, summarize


def _write_run(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _row(qid: str, p: float, r: float, f: float) -> dict:
    return {
        "id": qid,
        "faithfulness": {
            "citation_precision": p,
            "citation_recall": r,
            "citation_f1": f,
        },
    }


def test_load_run_skips_error_rows(tmp_path: Path):
    p = tmp_path / "results.jsonl"
    _write_run(p, [_row("q1", 1, 1, 1), {"id": "q2", "error": "boom"}])
    rows = load_run(str(p))
    assert [r["id"] for r in rows] == ["q1"]


def test_render_report_has_ci_table_and_significance(tmp_path: Path):
    ids = [f"q{i}" for i in range(10)]
    good = tmp_path / "good" / "results.jsonl"
    bad = tmp_path / "bad" / "results.jsonl"
    _write_run(good, [_row(i, 1.0, 1.0, 1.0) for i in ids])
    _write_run(bad, [_row(i, 0.0, 0.0, 0.0) for i in ids])

    s_good = summarize("good", load_run(str(good)), n_boot=500, seed=0)
    s_bad = summarize("bad", load_run(str(bad)), n_boot=500, seed=0)

    # Baseline = good; the bad config should be significantly worse.
    report = render_report([s_good, s_bad], baseline=s_good, n_boot=500, seed=0)
    assert "Per-configuration citation faithfulness" in report
    assert "`good`" in report and "`bad`" in report
    assert "Pairwise significance" in report
    # bad vs good: ΔF1 = -1.000 and the CI must exclude zero (marked with *).
    assert "-1.000" in report
    assert "*" in report


def test_summary_supported_rate(tmp_path: Path):
    p = tmp_path / "r" / "results.jsonl"
    _write_run(p, [_row("q1", 1, 1.0, 1), _row("q2", 1, 0.5, 0.6), _row("q3", 1, 1.0, 1)])
    s = summarize("r", load_run(str(p)), n_boot=200, seed=0)
    assert s.supported_rate == 2 / 3  # q1 and q3 fully supported (recall == 1.0)


def test_figure_series_shapes(tmp_path: Path):
    p = tmp_path / "r" / "results.jsonl"
    _write_run(p, [_row(f"q{i}", 0.8, 0.6, 0.68) for i in range(5)])
    s = summarize("r", load_run(str(p)), n_boot=200, seed=0)
    series = figure_series([s])
    assert {ser.metric for ser in series} == {
        "citation_precision",
        "citation_recall",
        "citation_f1",
    }
    for ser in series:
        assert len(ser.points) == 1
        assert ser.err_lo[0] >= 0 and ser.err_hi[0] >= 0
