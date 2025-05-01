from dataclasses import dataclass, field
from typing import Generic, Iterable, List, Self, Set, Tuple, TypeVar

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
    list: List[Tuple[_T, Set[int]]] = field(default_factory=list)

    @property
    def raw_list(self) -> List[_T]:
        return [msg for msg, _ in self.list]

    def to_original_indices(self, idx: int | Iterable[int]) -> Set[int]:
        return {
            orig_index
            for index in _to_set(idx)
            for orig_index in self.list[index][1]
        }

    def _get_remaining_indices(self) -> Set[int]:
        return {idx for (_, st) in self.list for idx in st}

    def get_removed_indices(self) -> Set[int]:
        return (
            set(range(self.start_index, self.end_index))
            - self._get_remaining_indices()
        )

    def __len__(self) -> int:
        return len(self.list)

    def select(self, idx: Iterable[int]) -> Self:
        return self.__class__(
            self.start_index, self.end_index, [self.list[i] for i in idx]
        )

    @classmethod
    def create(cls, list: List[_T], idx_offset: int = 0) -> Self:
        return cls(
            idx_offset,
            idx_offset + len(list),
            [(elem, {idx}) for idx, elem in enumerate(list, start=idx_offset)],
        )


def _to_set(idx: int | Iterable[int]) -> Set[int]:
    return {idx} if isinstance(idx, int) else set(idx)
