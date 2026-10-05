import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from aidial_adapter_vertexai.utils.env import get_env_int

T = TypeVar("T")
A = TypeVar("A")

# A single shared pool for all the blocking calls.
_THREAD_POOL = ThreadPoolExecutor(
    max_workers=get_env_int("THREAD_POOL_SIZE", 512)
)

_thread_lock = threading.Lock()


def _call_with_global_lock(func: Callable[[A], T], arg: A) -> T:
    with _thread_lock:
        return func(arg)


async def make_single_thread_async(func: Callable[[A], T], arg: A) -> T:
    """
    Function to run a synchronous function in separate thread,
    but only one at a time.
    """
    return await asyncio.to_thread(_call_with_global_lock, func, arg)


async def gather_sync(sync_tasks: list[Callable[[], T]]) -> list[T]:
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *(loop.run_in_executor(_THREAD_POOL, task) for task in sync_tasks)
    )
