"""ALCE-style citation faithfulness metrics.

Reproduces the citation precision / recall definitions from Gao et al.,
2023, *Enabling Large Language Models to Generate Text with Citations*
(EMNLP 2023) — the ALCE benchmark — and applies them to PaperPal's
``[paper_id:page]`` citation format.

Why this matters (and how it differs from PaperPal's built-in eval):
PaperPal's `backend/eval` measures whether the model cited the *right page
number* (set overlap of cited vs. gold pages). That answers "did it point
at the right place" but NOT "does the cited passage actually support the
claim." A model can cite the correct page and still state something the
page does not say. ALCE closes that gap with a Natural Language Inference
(NLI) model: a citation is only credited when the cited passage *entails*
the sentence it is attached to.

Definitions (following the official ALCE implementation):

    For a sentence ``s`` with a set of cited passages ``C = {c_1, ..., c_k}``:

    * Citation recall (per sentence) = 1 if the *concatenation* of all
      cited passages entails ``s`` (the sentence is fully supported),
      else 0. Sentences with no citation score 0.

    * Citation precision counts, over every individual citation ``c_j``,
      how many are *not irrelevant*. ``c_j`` is credited when either:
        (a) ``c_j`` alone entails ``s`` (it independently supports it), or
        (b) the full set entails ``s`` AND removing ``c_j`` breaks that
            support (``c_j`` is a necessary member of the group).
      A citation that neither independently supports nor is necessary is
      "irrelevant" (redundant or plain wrong) and is not credited.

Both metrics are *reference-free* for the citation itself — they need the
retrieved passage text and an NLI model, not a gold answer. That is what
makes them a faithfulness measure rather than a lexical-overlap proxy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .nli import NLIModel

# Matches PaperPal's inline citation format, e.g. [bdfaa68d8984f0dc:7].
# Kept byte-for-byte compatible with backend/eval/run_eval.py::CITATION_RE.
CITATION_RE = re.compile(r"\[([0-9a-fA-F]{6,32}):(\d+)\]")

# A citation key is (paper_id, page). Passages are looked up by this key.
CitationKey = tuple[str, int]


def split_sentences(text: str) -> list[str]:
    """Split answer text into sentences for statement-level scoring.

    A deliberately dependency-free splitter: breaks on ``.``/``!``/``?``
    terminators followed by whitespace. Citation markers like ``[id:7].``
    keep their trailing period attached to the sentence they close, so the
    citation travels with its statement. ALCE uses nltk's ``sent_tokenize``;
    swapping this out for nltk is a one-line change if exact parity is
    wanted (see ``SENTENCE_SPLITTER`` note in the README).
    """
    text = text.strip()
    if not text:
        return []
    # Split on sentence terminators followed by a space + likely start of a
    # new sentence. Keep the terminator with the preceding sentence.
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [s.strip() for s in raw if s.strip()]


def parse_citations(sentence: str) -> list[CitationKey]:
    """Return the ordered, de-duplicated ``(paper_id, page)`` citations in
    a sentence. Paper IDs are lower-cased so lookups are case-insensitive."""
    keys: list[CitationKey] = []
    seen: set[CitationKey] = set()
    for m in CITATION_RE.finditer(sentence):
        key = (m.group(1).lower(), int(m.group(2)))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def strip_citations(sentence: str) -> str:
    """Remove citation markers so the bare claim is the NLI hypothesis."""
    return CITATION_RE.sub("", sentence).strip()


def build_passage_map(retrieved: list[dict]) -> dict[CitationKey, str]:
    """Map each retrieved ``(paper_id, page)`` to its passage text.

    Multiple retrieved chunks can share a page; their texts are concatenated
    (de-duplicated) so a citation to that page sees everything the model was
    shown from it. A citation to a ``(paper_id, page)`` absent from this map
    was never in the model's context — a fabricated/hallucinated citation —
    and resolves to an empty passage, which cannot entail anything.
    """
    by_key: dict[CitationKey, list[str]] = {}
    for r in retrieved:
        key = (str(r["paper_id"]).lower(), int(r["page"]))
        text = str(r.get("text", "")).strip()
        bucket = by_key.setdefault(key, [])
        if text and text not in bucket:
            bucket.append(text)
    return {k: "\n\n".join(v) for k, v in by_key.items()}


@dataclass(frozen=True)
class SentenceScore:
    text: str
    citations: list[CitationKey]
    supported: bool  # recall contribution: full cited set entails the sentence
    n_citations: int
    n_precise: int  # citations credited toward precision
    hallucinated: list[CitationKey] = field(default_factory=list)  # cited but not retrieved


@dataclass(frozen=True)
class FaithfulnessScore:
    citation_recall: float
    citation_precision: float
    citation_f1: float
    n_sentences: int  # sentences considered in the recall denominator
    n_cited_sentences: int
    n_citations: int
    n_hallucinated: int
    sentences: list[SentenceScore] = field(default_factory=list)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score_answer(
    answer: str,
    retrieved: list[dict],
    nli: NLIModel,
    *,
    count_uncited_in_recall: bool = True,
) -> FaithfulnessScore:
    """Compute ALCE citation precision/recall/F1 for one answer.

    Args:
        answer: the model's answer text, containing ``[paper_id:page]`` markers.
        retrieved: the ``retrieved`` SSE payload — each item needs
            ``paper_id``, ``page``, ``text``.
        nli: an entailment model (``nli.entails(premise, hypothesis) -> bool``).
        count_uncited_in_recall: if True (ALCE default), sentences with no
            citation count as unsupported (recall 0) in the denominator. If
            False, only cited sentences enter the recall denominator — useful
            for isolating "when the model *does* cite, is it right?".

    Returns:
        A ``FaithfulnessScore`` with corpus-of-one aggregates plus per-sentence
        detail. Recall is the mean of per-sentence support; precision is the
        micro-average over individual citations (ALCE convention).
    """
    passages = build_passage_map(retrieved)
    sentences = split_sentences(answer)

    sentence_scores: list[SentenceScore] = []
    recall_hits = 0
    recall_denom = 0
    precise_total = 0
    citation_total = 0

    for sent in sentences:
        keys = parse_citations(sent)
        hypothesis = strip_citations(sent)
        if not hypothesis:
            continue  # a line that is only a citation marker — skip entirely

        if not keys:
            # Uncited statement. ALCE default: counts as unsupported.
            if count_uncited_in_recall:
                recall_denom += 1
            sentence_scores.append(
                SentenceScore(sent, [], supported=False, n_citations=0, n_precise=0)
            )
            continue

        recall_denom += 1
        cited_texts = [passages.get(k, "") for k in keys]
        hallucinated = [k for k in keys if not passages.get(k)]

        # --- Recall: does the whole cited set entail the sentence? ---
        joint_premise = "\n\n".join(t for t in cited_texts if t)
        supported = bool(joint_premise) and nli.entails(joint_premise, hypothesis)
        if supported:
            recall_hits += 1

        # --- Precision: credit each individually-supporting or necessary citation. ---
        n_precise = 0
        for j in range(len(keys)):
            citation_total += 1
            solo = cited_texts[j]
            if solo and nli.entails(solo, hypothesis):
                n_precise += 1  # (a) independently supports
                continue
            if supported:
                rest = "\n\n".join(t for i, t in enumerate(cited_texts) if i != j and t)
                if not (rest and nli.entails(rest, hypothesis)):
                    n_precise += 1  # (b) necessary member of an entailing group
        precise_total += n_precise

        sentence_scores.append(
            SentenceScore(
                text=sent,
                citations=keys,
                supported=supported,
                n_citations=len(keys),
                n_precise=n_precise,
                hallucinated=hallucinated,
            )
        )

    recall = recall_hits / recall_denom if recall_denom else 0.0
    precision = precise_total / citation_total if citation_total else 0.0
    return FaithfulnessScore(
        citation_recall=recall,
        citation_precision=precision,
        citation_f1=_f1(precision, recall),
        n_sentences=recall_denom,
        n_cited_sentences=sum(1 for s in sentence_scores if s.citations),
        n_citations=citation_total,
        n_hallucinated=sum(len(s.hallucinated) for s in sentence_scores),
        sentences=sentence_scores,
    )
