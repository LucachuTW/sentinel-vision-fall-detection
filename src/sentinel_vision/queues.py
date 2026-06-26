from __future__ import annotations

import asyncio
from collections.abc import Callable


class LatestValueQueue[T](asyncio.Queue[T]):
    """A bounded queue that drops the oldest item to protect live-stream latency."""

    def __init__(self, maxsize: int, on_drop: Callable[[], None] | None = None) -> None:
        super().__init__(maxsize=maxsize)
        self.dropped = 0
        self._on_drop = on_drop

    def put_latest(self, item: T) -> None:
        if self.full():
            try:
                self.get_nowait()
                self.task_done()
                self.dropped += 1
                if self._on_drop:
                    self._on_drop()
            except asyncio.QueueEmpty:
                pass
        self.put_nowait(item)
