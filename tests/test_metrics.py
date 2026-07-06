"""Unit tests pinning the ALCE citation precision/recall logic.

These use the deterministic KeywordNLI stub so the *metric* math is verified
without any model download. Each test isolates one ALCE case.
"""

from __future__ import annotations

from citeval.metrics import (
    build_passage_map,
    parse_citations,
    score_answer,
    split_sentences,
)
from citeval.nli import KeywordNLI

NLI = KeywordNLI()


def _passage(pid: str, page: int, text: str) -> dict:
    return {"paper_id": pid, "page": page, "text": text}


def test_split_sentences_keeps_citations_with_statement():
    text = "Self-attention replaces recurrence [aa11bb22:3]. It uses 8 heads [aa11bb22:5]."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert "[aa11bb22:3]" in sents[0]
    assert "[aa11bb22:5]" in sents[1]


def test_parse_citations_dedupes_and_lowercases():
    assert parse_citations("x [AA11BB22:3] y [aa11bb22:3] z [aa11bb22:5]") == [
        ("aa11bb22", 3),
        ("aa11bb22", 5),
    ]


def test_build_passage_map_concats_same_page_chunks():
    m = build_passage_map(
        [_passage("aa", 3, "first chunk"), _passage("aa", 3, "second chunk")]
    )
    assert m[("aa", 3)] == "first chunk\n\nsecond chunk"


def test_supported_and_precise_single_citation():
    retrieved = [_passage("aa11bb22", 3, "The cat sat on the mat quietly.")]
    answer = "The cat sat on the mat quietly [aa11bb22:3]."
    s = score_answer(answer, retrieved, NLI)
    assert s.citation_recall == 1.0
    assert s.citation_precision == 1.0
    assert s.citation_f1 == 1.0
    assert s.n_hallucinated == 0


def test_hallucinated_citation_is_unsupported_and_imprecise():
    # Cited page 9 was never retrieved → fabricated citation.
    retrieved = [_passage("aa11bb22", 3, "Something unrelated entirely.")]
    answer = "The model used a batch size of 32768 tokens [aa11bb22:9]."
    s = score_answer(answer, retrieved, NLI)
    assert s.citation_recall == 0.0
    assert s.citation_precision == 0.0
    assert s.n_hallucinated == 1
    assert s.sentences[0].hallucinated == [("aa11bb22", 9)]


def test_redundant_citation_not_credited():
    # Page 5 alone fully supports; page 6 (about something else) is redundant.
    retrieved = [
        _passage("abc123", 5, "Multi head attention attends to different subspaces jointly."),
        _passage("abc123", 6, "Positional encodings use sinusoids of different frequencies."),
    ]
    answer = "Multi head attention attends to different subspaces jointly [abc123:5][abc123:6]."
    s = score_answer(answer, retrieved, NLI)
    assert s.citation_recall == 1.0  # supported (page 5 covers it)
    assert s.citation_precision == 0.5  # 1 of 2 citations credited
    assert s.sentences[0].n_precise == 1
    assert s.sentences[0].n_citations == 2


def test_necessary_members_both_credited():
    # Neither passage alone entails; together they do → both are necessary.
    retrieved = [
        _passage("abc123", 1, "The alpha coefficient was measured."),
        _passage("abc123", 2, "The beta coefficient was measured."),
    ]
    answer = "Alpha beta [abc123:1][abc123:2]."
    s = score_answer(answer, retrieved, NLI)
    assert s.citation_recall == 1.0
    assert s.citation_precision == 1.0
    assert s.sentences[0].n_precise == 2


def test_uncited_sentence_counts_against_recall_by_default():
    retrieved = [_passage("abc123", 3, "The cat sat on the mat.")]
    answer = "The cat sat on the mat [abc123:3]. This extra claim has no citation."
    default = score_answer(answer, retrieved, NLI)
    assert default.n_sentences == 2
    assert default.citation_recall == 0.5  # 1 supported of 2

    # Toggling it off isolates "when it cites, is it right?"
    cited_only = score_answer(answer, retrieved, NLI, count_uncited_in_recall=False)
    assert cited_only.n_sentences == 1
    assert cited_only.citation_recall == 1.0
