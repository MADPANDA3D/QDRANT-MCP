from __future__ import annotations

import asyncio

import pytest

from mcp_server_qdrant.async_compat import _timeout_compat, timeout


@pytest.mark.asyncio
async def test_timeout_allows_work_before_deadline() -> None:
    async with timeout(1):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_python_310_timeout_shim_raises_timeout_error() -> None:
    with pytest.raises(TimeoutError):
        async with _timeout_compat(0.001):
            await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_python_310_timeout_shim_preserves_external_cancellation() -> None:
    entered = asyncio.Event()

    async def wait_for_cancellation() -> None:
        async with _timeout_compat(60):
            entered.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(wait_for_cancellation())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
