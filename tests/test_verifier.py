"""Tests for the NLI citation verifier (KeywordNLI, offline)."""

from __future__ import annotations

from citeval.nli import KeywordNLI
from citeval.verifier import verify_answer

NLI = KeywordNLI()


def _p(pid, page, text):
    return {"paper_id": pid, "page": page, "text": text}


RETRIEVED = [
    _p("abc123", 2, "The Transformer relies entirely on self-attention."),
    _p("abc123", 10, "References and acknowledgements section."),
]


def test_flags_unsupported_and_repairs_to_best_page():
    # Model cited page 10 (references) for a self-attention claim on page 2.
    answer = "The Transformer relies entirely on self-attention [abc123:10]."
    (v,) = verify_answer(answer, RETRIEVED, NLI, threshold=0.5)
    assert v.original == [("abc123", 10)]
    assert v.original_supported is False  # page 10 doesn't entail
    assert v.repaired == ("abc123", 2)  # re-attributed to the entailing page
    assert v.abstained is False


def test_supported_citation_stays():
    answer = "The Transformer relies entirely on self-attention [abc123:2]."
    (v,) = verify_answer(answer, RETRIEVED, NLI, threshold=0.5)
    assert v.original_supported is True
    assert v.repaired == ("abc123", 2)


def test_abstains_when_nothing_entails():
    # A claim no retrieved passage supports → verifier declines to cite.
    answer = "The model was trained on 4.5 million sentence pairs [abc123:2]."
    (v,) = verify_answer(answer, RETRIEVED, NLI, threshold=0.5)
    assert v.abstained is True
    assert v.repaired is None


def test_uncited_sentences_are_not_verified():
    answer = "Self-attention is the core mechanism. It has no citation here."
    assert verify_answer(answer, RETRIEVED, NLI) == []
