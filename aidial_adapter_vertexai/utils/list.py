from abc import ABC, abstractmethod
from collections.abc import Callable, Container
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def omit_by_indices(lst: list[_T], indices: Container[int]) -> list[_T]:
    return [elem for idx, elem in enumerate(lst) if idx not in indices]


def group_by(
    lst: list[_T],
    key: Callable[[_T], Any],
    init: Callable[[_T], _V],
    merge: Callable[[_V, _T], _V],
) -> list[_V]:
    def _gen():
        if not lst:
            return

        prev_val = init(lst[0])
        prev_key = key(lst[0])

        for elem in lst[1:]:
            if prev_key == key(elem):
                prev_val = merge(prev_val, elem)
            else:
                yield prev_val
                prev_val = init(elem)
                prev_key = key(elem)

        yield prev_val

    return list(_gen())


class MessageMergeStrategy(Generic[_T], ABC):
    @staticmethod
    @abstractmethod
    def role(_T) -> Any: ...

    @staticmethod
    @abstractmethod
    def merge(a: _T, b: _T) -> _T: ...
