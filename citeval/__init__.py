"""cite-faithfulness — NLI-based citation faithfulness evaluation for RAG.

Reproduces the ALCE (Gao et al., 2023) citation precision/recall metric and
applies it to citation-grounded RAG systems (default target: PaperPal).
"""

from .metrics import FaithfulnessScore, SentenceScore, score_answer
from .nli import CrossEncoderNLI, KeywordNLI, MockNLI, NLIModel, get_nli
from .stats import CI, DiffResult, McNemarResult, bootstrap_ci, mcnemar_exact, paired_diff

__all__ = [
    "CI",
    "CrossEncoderNLI",
    "DiffResult",
    "FaithfulnessScore",
    "KeywordNLI",
    "McNemarResult",
    "MockNLI",
    "NLIModel",
    "SentenceScore",
    "bootstrap_ci",
    "get_nli",
    "mcnemar_exact",
    "paired_diff",
    "score_answer",
]
