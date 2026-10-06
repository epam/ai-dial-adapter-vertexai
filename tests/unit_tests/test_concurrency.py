import asyncio
import time

import pytest

from aidial_adapter_vertexai.utils.concurrency import gather_sync


async def test_gather_sync_preserves_order():
    assert await gather_sync([lambda: 1, lambda: 2, lambda: 3]) == [1, 2, 3]


async def test_gather_sync_propagates_errors():
    with pytest.raises(ValueError, match="boom"):
        await gather_sync([lambda: (_ for _ in ()).throw(ValueError("boom"))])


async def test_cancellation_does_not_block_the_event_loop():
    """
    A `gather_sync` cancelled while its tasks are still running must not stall
    the event loop: a pool per call used to join the workers on the loop
    thread, freezing `/health` for as long as the blocking calls took.
    """
    blocking = 2.0
    task = asyncio.create_task(gather_sync([lambda: time.sleep(blocking)]))
    await asyncio.sleep(0.1)

    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The loop must keep turning while the abandoned thread runs on.
    await asyncio.sleep(0)
    elapsed = time.perf_counter() - started

    assert elapsed < blocking / 4, (
        f"the event loop was blocked for {elapsed:.2f}s by a cancelled call"
    )
