"""Self-contained demo: ALCE citation scoring with no server and no downloads.

Runs the faithfulness metric on a hand-built example using the deterministic
KeywordNLI stub, so the whole pipeline (segmentation → citation parsing →
passage lookup → precision/recall) is exercisable in one command:

    python -m citeval.demo

The example is engineered so every ALCE case appears exactly once:
a supported+precise citation, a hallucinated (never-retrieved) citation, a
redundant citation, and an uncited claim. Expected output is asserted in
tests/test_demo.py so this stays a living, checkable worked example.
"""

from __future__ import annotations

from .metrics import score_answer
from .nli import KeywordNLI

# A retrieval payload shaped exactly like PaperPal's `retrieved` SSE event.
RETRIEVED = [
    {
        "paper_id": "aaaa1111bbbb2222",
        "page": 3,
        "text": "The Transformer relies entirely on self-attention to compute "
        "representations without recurrence or convolution.",
    },
    {
        "paper_id": "aaaa1111bbbb2222",
        "page": 5,
        "text": "Multi-head attention lets the model jointly attend to information "
        "from different representation subspaces at different positions.",
    },
    {
        "paper_id": "aaaa1111bbbb2222",
        "page": 6,
        "text": "Sinusoidal positional encodings are added to the input embeddings "
        "so the model can use the order of the sequence.",
    },
]

# Answer exhibits: supported+precise cite [.:3]; a redundant extra cite [.:6]
# on a claim already fully supported by [.:5]; a hallucinated cite [.:9] to a
# page that was never retrieved; and a final uncited claim.
ANSWER = (
    "The Transformer relies entirely on self-attention without recurrence or "
    "convolution [aaaa1111bbbb2222:3]. Multi-head attention lets the model jointly "
    "attend to information from different representation subspaces at different "
    "positions [aaaa1111bbbb2222:5][aaaa1111bbbb2222:6]. The model was trained on "
    "the WMT 2014 English-German dataset [aaaa1111bbbb2222:9]. It achieves strong "
    "translation quality."
)


def run() -> None:
    score = score_answer(ANSWER, RETRIEVED, KeywordNLI())
    print("=== citeval demo (KeywordNLI stub — illustrative, not real NLI) ===")
    print(f"citation precision : {score.citation_precision:.3f}")
    print(f"citation recall    : {score.citation_recall:.3f}")
    print(f"citation F1        : {score.citation_f1:.3f}")
    print(f"sentences (recall denom) : {score.n_sentences}")
    print(f"citations total          : {score.n_citations}")
    print(f"hallucinated citations   : {score.n_hallucinated}")
    print()
    for i, s in enumerate(score.sentences, 1):
        cites = ", ".join(f"{p}:{pg}" for p, pg in s.citations) or "—"
        flag = " HALLUC" if s.hallucinated else ""
        print(
            f"[{i}] supported={str(s.supported):5} "
            f"precise={s.n_precise}/{s.n_citations} cites=[{cites}]{flag}"
        )
        print(f"     {s.text}")


if __name__ == "__main__":
    run()
