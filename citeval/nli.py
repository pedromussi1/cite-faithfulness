"""Natural Language Inference backends for citation faithfulness.

A citation is credited only when its passage *entails* the cited sentence.
The entailment judgement is delegated to an ``NLIModel``:

    entails(premise, hypothesis) -> bool

Two backends ship here:

* ``CrossEncoderNLI`` — the real judge. A local sentence-transformers
  cross-encoder trained on (M)NLI. Runs on CPU or the RTX 4070 Ti, $0, no
  API. ALCE used a large TRUE/T5-11B NLI model; we use a small DeBERTa-MNLI
  checkpoint that fits 12 GB and downloads once. The exact checkpoint is a
  pinned config knob so the reproduction is auditable.

* ``KeywordNLI`` (a.k.a. MockNLI) — a deterministic, dependency-free stand-in
  used by the unit tests and CI. It calls "entailed" when the hypothesis's
  content words are a subset of the premise's. This has no ML in it; its only
  job is to exercise and pin the *metric* logic (precision/recall aggregation,
  the "necessary member" rule, hallucinated-citation handling) without
  downloading a model. Never use it for real numbers.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class NLIModel(Protocol):
    """Anything that can judge whether ``premise`` entails ``hypothesis``."""

    def entails(self, premise: str, hypothesis: str) -> bool: ...


_WORD_RE = re.compile(r"[a-z0-9]+")
# Content-word filter for the keyword stub: ignore common function words so
# "supported" doesn't hinge on matching "the"/"of"/etc.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "they", "this", "to", "was", "were", "which", "with", "without",
})


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


class KeywordNLI:
    """Deterministic entailment stub for tests/CI — NOT a real NLI model.

    ``entails`` is True when at least ``coverage`` of the hypothesis's content
    words appear in the premise. Empty premise never entails (models a
    fabricated citation whose passage was never retrieved).
    """

    def __init__(self, coverage: float = 0.999) -> None:
        self._coverage = coverage

    def entails(self, premise: str, hypothesis: str) -> bool:
        if not premise.strip():
            return False
        hyp = _content_words(hypothesis)
        if not hyp:
            return True
        prem = _content_words(premise)
        overlap = len(hyp & prem) / len(hyp)
        return overlap >= self._coverage


# Backwards/intent-friendly alias used in tests.
MockNLI = KeywordNLI


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_plain_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT_RE.split(text.strip())) if s]


def pack_windows(text: str, *, max_chars: int, overlap_sents: int) -> list[str]:
    """Split ``text`` into overlapping sentence windows each <= ~max_chars.

    Small NLI cross-encoders truncate at 512 tokens, so a fact near the end of
    a long passage gets silently cut and misjudged as unsupported. Scoring the
    hypothesis against each window and taking the max entailment makes the
    verdict independent of where in the passage the support lives. Consecutive
    windows share ``overlap_sents`` sentences so support spanning a boundary
    isn't lost.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = _split_plain_sentences(text)
    if not sentences:
        return [text[:max_chars]]
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + len(s) + 1 > max_chars:
            windows.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        windows.append(" ".join(cur))
    return windows


class CrossEncoderNLI:
    """Real NLI judge: a local cross-encoder trained on (M)NLI.

    Lazy-loads the model on first use so importing this module (e.g. in CI or
    for the metric unit tests) never pulls in torch. ``entails`` returns True
    when the model's ``entailment`` probability clears ``threshold``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        threshold: float = 0.5,
        device: str | None = None,
        window_chars: int = 1200,
        overlap_sents: int = 1,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._device = device
        # ~1200 chars keeps premise + hypothesis under the model's 512-token
        # limit (roughly 4 chars/token) with headroom for the hypothesis.
        self._window_chars = window_chars
        self._overlap_sents = overlap_sents
        self._model = None  # loaded on first entails()
        self._entail_idx: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder  # heavy; imported lazily

        self._model = CrossEncoder(self.model_name, device=self._device)
        # cross-encoder/nli-* models emit labels in the order
        # ["contradiction", "entailment", "neutral"]; resolve the entailment
        # column from the model config rather than hard-coding an index.
        id2label = getattr(self._model.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if str(label).lower().startswith("entail"):
                self._entail_idx = int(idx)
                break
        if self._entail_idx is None:
            self._entail_idx = 1  # documented default for cross-encoder/nli-*

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """Max entailment probability of ``hypothesis`` over the premise's
        windows. Windowing makes the score independent of where the support
        sits in a long passage (see ``pack_windows``)."""
        if not premise.strip():
            return 0.0
        self._ensure_loaded()
        assert self._model is not None and self._entail_idx is not None
        import numpy as np

        windows = pack_windows(
            premise, max_chars=self._window_chars, overlap_sents=self._overlap_sents
        )
        pairs = [(w, hypothesis) for w in windows]
        scores = np.asarray(self._model.predict(pairs, apply_softmax=True))
        return float(scores[:, self._entail_idx].max())

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.entail_prob(premise, hypothesis) >= self.threshold


def get_nli(name: str, **kwargs) -> NLIModel:
    """Factory: ``"mock"`` → KeywordNLI, anything else → CrossEncoderNLI(name)."""
    if name in {"mock", "keyword"}:
        return KeywordNLI(**kwargs)
    return CrossEncoderNLI(name, **kwargs)
