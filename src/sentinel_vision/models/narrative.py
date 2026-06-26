from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from sentinel_vision.config import OllamaConfig
from sentinel_vision.domain import PipelineEvent

LOGGER = logging.getLogger(__name__)
_SEVERITY = {"info": 0, "warning": 1, "critical": 2}


class LocalNarrativeService:
    """Optional local-LLM enrichment isolated from the real-time decision path."""

    def __init__(
        self,
        config: OllamaConfig,
        on_narrative: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self.config = config
        self._on_narrative = on_narrative
        self._queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=32)
        self._stop = asyncio.Event()

    @property
    def backend_name(self) -> str:
        return f"ollama:{self.config.model}" if self.config.enabled else "disabled"

    def submit(self, event: PipelineEvent) -> None:
        if not self.config.enabled:
            return
        if _SEVERITY[event.severity] < _SEVERITY[self.config.min_severity]:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(event)

    async def run(self) -> None:
        if not self.config.enabled:
            return
        async with httpx.AsyncClient(
            base_url=self.config.endpoint,
            timeout=self.config.timeout_seconds,
        ) as client:
            while not self._stop.is_set():
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                try:
                    narrative = await self._generate(client, event)
                    await self._on_narrative(event.event_id, narrative)
                except Exception:
                    LOGGER.warning("local narrative generation failed", exc_info=True)
                finally:
                    self._queue.task_done()

    async def _generate(self, client: httpx.AsyncClient, event: PipelineEvent) -> str:
        prompt = (
            "You write terse industrial monitoring incident notes. Use only the supplied facts; "
            "do not diagnose or invent causes. Return one sentence in Spanish, maximum 25 words. "
            f"Facts: camera={event.source_id}, track={event.track_id}, action={event.action}, "
            f"confidence={event.confidence:.2f}, severity={event.severity}."
        )
        response = await client.post(
            "/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 64},
            },
        )
        response.raise_for_status()
        text = str(response.json().get("response", "")).strip()
        if not text:
            raise ValueError("Ollama returned an empty narrative")
        return " ".join(text.split())[:300]

    def stop(self) -> None:
        self._stop.set()
