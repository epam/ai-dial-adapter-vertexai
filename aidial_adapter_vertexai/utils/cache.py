import asyncio
import json
from typing import Callable, Coroutine, Generic, ParamSpec, Protocol, TypeVar

from aidial_adapter_vertexai.utils.log_config import app_logger as log

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)


class _CachedFunction(Protocol, Generic[_P, _R]):
    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    async def clear(self): ...


def cache(
    close: Callable[[_R], Coroutine[None, None, None]] | None = None,
) -> Callable[
    [Callable[_P, Coroutine[None, None, _R]]], _CachedFunction[_P, _R]
]:
    def wrapper(
        f: Callable[_P, Coroutine[None, None, _R]],
    ) -> _CachedFunction[_P, _R]:
        class wrapped:
            _tasks: dict[str, asyncio.Task[_R]]
            _lock: asyncio.Lock

            def __init__(self) -> None:
                self._tasks = {}
                self._lock = asyncio.Lock()

            async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
                key = json.dumps(
                    {"args": args, "kwargs": kwargs}, sort_keys=True
                )

                async with self._lock:
                    if (task := self._tasks.get(key)) is None:
                        task = self._tasks[key] = asyncio.create_task(
                            f(*args, **kwargs)
                        )

                try:
                    return await task
                except Exception:
                    async with self._lock:
                        if self._tasks.get(key) is task:
                            del self._tasks[key]
                    raise

            async def clear(self):
                async with self._lock:
                    entries = self._tasks
                    self._tasks = {}

                func_name = f"{f.__module__}.{f.__qualname__}"
                log.debug(f"Clearing cache {func_name}")

                for key, task in entries.items():
                    try:
                        value = task.result()
                    except Exception:
                        continue

                    log.debug(f"Closing cached value {func_name}({key})")

                    try:
                        if close is not None:
                            await close(value)
                    except Exception as e:
                        log.error(f"Error on closing the task: {e}")

        return wrapped()

    return wrapper
