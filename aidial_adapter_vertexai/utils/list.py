from typing import Any, Callable, List, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def group_by(
    lst: List[_T],
    key: Callable[[_T], Any],
    init: Callable[[_T], _V],
    merge: Callable[[_V, _T], _V],
) -> List[_V]:

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
