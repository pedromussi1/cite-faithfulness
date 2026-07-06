"""Tests for offline re-scoring (no server/model)."""

from __future__ import annotations

from citeval.nli import KeywordNLI
from citeval.rescore import rescore_row

NLI = KeywordNLI()


def _row(**over):
    base = {
        "id": "q1",
        "question": "?",
        "answer": "The cat sat on the mat [abc123:3].",
        "retrieved": [
            {"paper_id": "abc123", "page": 3, "text": "The cat sat on the mat quietly."}
        ],
        "faithfulness": {"citation_f1": 0.0},  # stale score to be overwritten
        "sentences": [],
    }
    base.update(over)
    return base


def test_rescore_recomputes_faithfulness():
    out = rescore_row(_row(), NLI)
    assert out is not None
    assert out["faithfulness"]["citation_precision"] == 1.0
    assert out["faithfulness"]["citation_recall"] == 1.0
    assert out["faithfulness"]["citation_f1"] == 1.0
    assert "rescored_ts" in out
    # original non-score fields are preserved
    assert out["id"] == "q1" and out["question"] == "?"


def test_rescore_passthrough_error_rows():
    err = {"id": "q2", "error": "boom"}
    assert rescore_row(err, NLI) == err


def test_rescore_skips_rows_without_stored_passages():
    old = {"id": "q3", "answer": "x [abc123:3].", "retrieved_keys": ["abc123:3"]}
    assert rescore_row(old, NLI) is None
