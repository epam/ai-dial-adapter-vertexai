import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

T = TypeVar("T")
A = TypeVar("A")

_single_thread_async_lock = asyncio.Lock()


async def make_single_thread_async(func: Callable[[A], T], arg: A) -> T:
    """
    Function to run a synchronous function in separate thread,
    but only one at a time.
    """
    async with _single_thread_async_lock:
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, func, arg)


async def make_async(func: Callable[[A], T], arg: A) -> T:
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, arg)


async def gather_sync(sync_tasks: List[Callable[[], T]], **kwargs) -> List[T]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(**kwargs) as executor:
        tasks = [loop.run_in_executor(executor, task) for task in sync_tasks]
        return await asyncio.gather(*tasks)
