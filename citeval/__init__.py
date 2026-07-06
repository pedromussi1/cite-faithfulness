"""cite-faithfulness — NLI-based citation faithfulness evaluation for RAG.

Reproduces the ALCE (Gao et al., 2023) citation precision/recall metric and
applies it to citation-grounded RAG systems (default target: PaperPal).
"""

from .metrics import FaithfulnessScore, SentenceScore, score_answer
from .nli import CrossEncoderNLI, KeywordNLI, MockNLI, NLIModel, get_nli

__all__ = [
    "CrossEncoderNLI",
    "FaithfulnessScore",
    "KeywordNLI",
    "MockNLI",
    "NLIModel",
    "SentenceScore",
    "get_nli",
    "score_answer",
]
