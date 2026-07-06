"""Async client for a running PaperPal backend.

The faithfulness study treats PaperPal as the *system under test* and drives
it over its public HTTP API — exactly as ``PaperPal/backend/eval/run_eval.py``
does — rather than importing its internals. That keeps this repo decoupled
from PaperPal's code: any RAG server that exposes the same ``/query`` SSE
contract (``retrieved`` → ``token`` → ``done``) can be evaluated by swapping
``--base-url``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class QueryOutcome:
    answer: str
    retrieved: list[dict[str, Any]]  # each: paper_id, page, chunk_idx, text, score


async def _parse_sse(stream: AsyncIterator[str]) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``(event_type, parsed_data)`` from an SSE text stream.

    Ported from PaperPal's eval harness so parsing stays identical.
    """
    buffer = ""
    async for chunk in stream:
        buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event_type = "message"
            data_lines: list[str] = []
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                yield event_type, json.loads("\n".join(data_lines))


class PaperPalClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 180.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> PaperPalClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def healthz(self) -> bool:
        try:
            r = await self._client.get(f"{self._base_url}/healthz")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_uploaded(self, pdf_dir: Path) -> dict[str, str]:
        """Upload every PDF in ``pdf_dir`` not already indexed.

        Returns ``{filename: paper_id}``. paper_id is the first 16 hex chars of
        the file's SHA-256, matching PaperPal's ingest id derivation.
        """
        r = await self._client.get(f"{self._base_url}/docs/list")
        r.raise_for_status()
        existing = {d["paper_id"] for d in r.json().get("documents", [])}

        mapping: dict[str, str] = {}
        for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
            data = pdf.read_bytes()
            pid = hashlib.sha256(data).hexdigest()[:16]
            mapping[pdf.name] = pid
            if pid in existing:
                continue
            files = {"file": (pdf.name, data, "application/pdf")}
            resp = await self._client.post(f"{self._base_url}/upload", files=files)
            resp.raise_for_status()
        return mapping

    async def query(
        self,
        question: str,
        *,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> QueryOutcome:
        payload = {"question": question, "paper_ids": paper_ids, "top_k": top_k}
        retrieved: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        final_answer: str | None = None
        async with self._client.stream(
            "POST", f"{self._base_url}/query", json=payload
        ) as resp:
            resp.raise_for_status()
            async for event_type, data in _parse_sse(resp.aiter_text()):
                if event_type == "retrieved":
                    retrieved = data
                elif event_type == "token":
                    answer_parts.append(data["text"])
                elif event_type == "done":
                    final_answer = data["answer"]
                elif event_type == "error":
                    raise RuntimeError(f"Stream error: {data.get('message')}")
        return QueryOutcome(answer=final_answer or "".join(answer_parts), retrieved=retrieved)
