from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run with uvloop when available; fall back to the standard event loop."""
    try:
        import uvloop
    except ImportError:  # pragma: no cover - platform dependent
        return asyncio.run(coroutine)
    return uvloop.run(coroutine)
