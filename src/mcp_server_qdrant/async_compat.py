"""Small asyncio compatibility helpers for the declared Python support range."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def _timeout_compat(delay: float | None) -> AsyncIterator[None]:
    """Backport the essential ``asyncio.timeout`` behavior for Python 3.10."""

    if delay is None:
        yield
        return

    task = asyncio.current_task()
    if task is None:  # pragma: no cover - an async context always has a task
        raise RuntimeError("timeout context requires a running asyncio task")

    loop = asyncio.get_running_loop()
    expired = False

    def cancel_for_timeout() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    handle = loop.call_at(loop.time() + max(0.0, float(delay)), cancel_for_timeout)
    try:
        yield
    except asyncio.CancelledError as exc:
        if expired:
            raise TimeoutError from exc
        raise
    finally:
        handle.cancel()


@asynccontextmanager
async def timeout(delay: float | None) -> AsyncIterator[None]:
    """Use the native timeout context when available, otherwise the 3.10 shim."""

    native_timeout = getattr(asyncio, "timeout", None)
    if callable(native_timeout):
        async with native_timeout(delay):
            yield
        return

    async with _timeout_compat(delay):
        yield
