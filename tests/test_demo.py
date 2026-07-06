"""Pin the worked example in citeval.demo so it stays a checkable artifact."""

from __future__ import annotations

from citeval.demo import ANSWER, RETRIEVED
from citeval.metrics import score_answer
from citeval.nli import KeywordNLI


def test_demo_expected_scores():
    s = score_answer(ANSWER, RETRIEVED, KeywordNLI())
    # Hand-traced in demo.py: S1 supported+precise, S2 one redundant cite,
    # S3 hallucinated cite, S4 uncited claim.
    assert s.n_sentences == 4
    assert s.n_citations == 4
    assert s.n_hallucinated == 1
    assert s.citation_recall == 0.5
    assert s.citation_precision == 0.5
    assert s.citation_f1 == 0.5
