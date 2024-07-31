import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

T = TypeVar("T")
A = TypeVar("A")


async def make_async(func: Callable[[A], T], arg: A) -> T:
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, arg)


async def gather_sync(sync_tasks: List[Callable[[], T]], **kwargs) -> List[T]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(**kwargs) as executor:
        tasks = [loop.run_in_executor(executor, task) for task in sync_tasks]
        return await asyncio.gather(*tasks)
