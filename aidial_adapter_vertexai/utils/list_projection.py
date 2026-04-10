import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, Self, TypeVar

_T = TypeVar("_T")


@dataclass
class ListProjection(Generic[_T]):
    """
    The class represents a transformation of the original list which may
    include removal of the elements of the original list and
    merging of the subsequent elements of the original list.

    Each derivative element is mapped onto a subset of the original elements.
    The subsets are disjoint.
    """

    start_index: int
    end_index: int
    lst: list[tuple[_T, set[int]]] = field(default_factory=list)

    @property
    def raw_list(self) -> builtins.list[_T]:
        return [msg for msg, _ in self.lst]

    def _get_remaining_indices(self) -> set[int]:
        return {idx for (_, st) in self.lst for idx in st}

    def get_removed_indices(self) -> set[int]:
        return (
            set(range(self.start_index, self.end_index))
            - self._get_remaining_indices()
        )

    def __len__(self) -> int:
        return len(self.lst)

    def select(self, idx: Iterable[int]) -> Self:
        return self.__class__(
            self.start_index, self.end_index, [self.lst[i] for i in idx]
        )

    @classmethod
    def create(cls, list: builtins.list[_T], idx_offset: int = 0) -> Self:
        return cls(
            idx_offset,
            idx_offset + len(list),
            [(elem, {idx}) for idx, elem in enumerate(list, start=idx_offset)],
        )
