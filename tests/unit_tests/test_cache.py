import asyncio
import time

from aidial_adapter_vertexai.utils.cache import cache
from aidial_adapter_vertexai.utils.concurrency import make_single_thread_async


@cache()
async def _long_running_task(key: str) -> int:
    def _task(_dummy) -> int:
        time.sleep(2)
        return 10

    return await make_single_thread_async(_task, ())


async def test_concurrent_fetches_to_the_same_cache():
    n = 10
    results = await asyncio.gather(*[_long_running_task("x") for _ in range(n)])
    assert len(results) == n
