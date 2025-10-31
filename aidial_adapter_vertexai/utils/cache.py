import json
from threading import Lock
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
)

from aidial_adapter_vertexai.utils.log_config import app_logger as log

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)


class CachedFunction(Protocol, Generic[_P, _R]):
    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...

    async def clear(self): ...


def cache(
    close: Callable[[_R], Coroutine[Any, Any, None]] | None = None,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], CachedFunction[_P, _R]]:

    def wrapper(
        f: Callable[_P, Coroutine[Any, Any, _R]],
    ) -> CachedFunction[_P, _R]:
        class wrapped:
            entries: dict[str, _R]
            lock: Lock

            def __init__(self) -> None:
                self.entries = {}
                self.lock = Lock()

            async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
                key = json.dumps(
                    {"args": args, "kwargs": kwargs}, sort_keys=True
                )

                with self.lock:
                    if key not in self.entries:
                        self.entries[key] = await f(*args, **kwargs)
                    return self.entries[key]

            async def clear(self):
                with self.lock:
                    entries = self.entries
                    self.entries = {}

                for key, value in entries.items():
                    log.debug(
                        f"Closing cached value ({value}) corresponding to the key {key}."
                    )
                    if close:
                        try:
                            await close(value)
                        except Exception as e:
                            log.error(f"Error on closing: {e}")

        return wrapped()

    return wrapper
