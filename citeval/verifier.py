"""NLI citation verifier — the study's novel contribution.

A RAG model emits citations `[paper_id:page]` that it *believes* support each
sentence. This verifier applies a cheap, local NLI model to those citations to
do two things:

* **Flag** — for each cited sentence, decide whether its *own* citations
  actually entail the claim. Unsupported sentences can be surfaced to the user
  ("this citation may not support the statement") instead of silently trusted.

* **Repair** — independently of what the model cited, search the *retrieved
  pool* for the passage that best entails the claim, and re-attribute the
  citation to it. If nothing in the pool entails the claim above a threshold,
  the verifier *abstains* (drops the citation and marks the sentence
  unsupported) rather than assert a citation it can't justify.

Design note on evaluation: because repair uses NLI, scoring the repaired
answer with the *same* NLI would be circular (trivially ~perfect). The honest
evaluation (see `verify_eval.py`) measures the verifier against **independent
ground truth** — the human-curated gold pages, and optional human statement
labels — never against its own judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import (
    CitationKey,
    build_passage_map,
    parse_citations,
    split_sentences,
    strip_citations,
)
from .nli import NLIModel


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    hypothesis: str  # sentence with citation markers stripped
    original: list[CitationKey]  # what the model cited
    original_supported: bool  # do the model's own citations entail the claim?
    repaired: CitationKey | None  # best-entailing retrieved passage (None = abstain)
    repaired_prob: float  # entailment score of the best passage
    abstained: bool  # True when no retrieved passage entails above threshold

    @property
    def original_pages(self) -> set[int]:
        return {page for _, page in self.original}


def verify_answer(
    answer: str,
    retrieved: list[dict],
    nli: NLIModel,
    *,
    threshold: float = 0.5,
) -> list[SentenceVerdict]:
    """Flag + repair every cited sentence in ``answer``.

    Only sentences that carry at least one citation are verified (an uncited
    sentence makes no citation claim to check). For each, we record whether the
    model's own citations entail it, and the best-entailing passage from the
    full retrieved pool as the repair target.
    """
    passages = build_passage_map(retrieved)
    pool = list(passages.items())  # [((paper_id, page), text), ...]

    verdicts: list[SentenceVerdict] = []
    for sent in split_sentences(answer):
        keys = parse_citations(sent)
        if not keys:
            continue  # no citation claim to verify
        hyp = strip_citations(sent)
        if not hyp:
            continue

        # Flag: do the model's *own* cited passages jointly entail the claim?
        cited_joint = "\n\n".join(t for k in keys if (t := passages.get(k, "")))
        original_supported = bool(cited_joint) and nli.entails(cited_joint, hyp)

        # Repair: best-entailing passage anywhere in the retrieved pool,
        # scored in a single batched forward pass.
        best_key: CitationKey | None = None
        best_prob = 0.0
        if pool:
            probs = nli.entail_probs([(text, hyp) for _, text in pool])
            best_i = max(range(len(probs)), key=lambda i: probs[i])
            best_prob, best_key = probs[best_i], pool[best_i][0]

        abstained = best_prob < threshold
        verdicts.append(
            SentenceVerdict(
                sentence=sent,
                hypothesis=hyp,
                original=keys,
                original_supported=original_supported,
                repaired=None if abstained else best_key,
                repaired_prob=best_prob,
                abstained=abstained,
            )
        )
    return verdicts
