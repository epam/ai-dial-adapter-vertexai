from typing import List, Set, TypeVar

_T = TypeVar("_T")


def select_by_indices(lst: List[_T], indices: Set[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx in indices]


def omit_by_indices(lst: List[_T], indices: Set[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx not in indices]
