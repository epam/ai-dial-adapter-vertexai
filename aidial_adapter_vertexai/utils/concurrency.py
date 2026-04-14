import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
A = TypeVar("A")

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


async def gather_sync(sync_tasks: list[Callable[[], T]], **kwargs) -> list[T]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(**kwargs) as executor:
        tasks = [loop.run_in_executor(executor, task) for task in sync_tasks]
        return await asyncio.gather(*tasks)
