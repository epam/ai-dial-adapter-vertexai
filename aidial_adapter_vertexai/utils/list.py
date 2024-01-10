from typing import Any, Callable, List, TypeVar

T = TypeVar("T")


def cluster_by(feature: Callable[[T], Any], xs: List[T]) -> List[List[T]]:
    if not xs:
        return []

    ret: List[List[T]] = [[xs[0]]]

    for x in xs[1:]:
        if feature(x) == feature(ret[-1][0]):
            ret[-1].append(x)
        else:
            ret.append([x])

    return ret
