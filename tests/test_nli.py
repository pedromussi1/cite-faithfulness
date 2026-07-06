"""Tests for NLI helpers (the pure windowing logic and the KeywordNLI stub)."""

from __future__ import annotations

from citeval.nli import KeywordNLI, pack_windows


def test_pack_windows_short_text_single_window():
    assert pack_windows("One short sentence.", max_chars=1200, overlap_sents=1) == [
        "One short sentence."
    ]


def test_pack_windows_empty():
    assert pack_windows("   ", max_chars=100, overlap_sents=1) == []


def test_pack_windows_splits_long_text_within_budget():
    text = " ".join(f"Sentence number {i} here." for i in range(50))
    windows = pack_windows(text, max_chars=100, overlap_sents=1)
    assert len(windows) > 1
    # Each window respects the budget (allowing a single over-long sentence).
    for w in windows:
        assert len(w) <= 100 or w.count(".") <= 1


def test_pack_windows_overlap_shares_a_sentence():
    text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five. Zeta six."
    windows = pack_windows(text, max_chars=30, overlap_sents=1)
    # Consecutive windows should share a boundary sentence (overlap).
    assert len(windows) >= 2
    first_last = windows[0].split(". ")[-1]
    assert first_last.rstrip(".") in windows[1]


def test_keyword_nli_still_works_for_metric_tests():
    nli = KeywordNLI()
    assert nli.entails("the cat sat on the mat", "cat sat mat")
    assert not nli.entails("", "anything")
