from typing import Any, List


def match_objects(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        assert list(sorted(expected.keys())) == list(sorted(actual.keys()))
        for k, v in expected.items():
            match_objects(v, actual[k])
    elif isinstance(expected, tuple):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            match_objects(expected[i], actual[i])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            match_objects(expected[i], actual[i])
    elif callable(expected):
        assert expected(
            actual
        ), f"The predicate failed on the actual result: {actual}"
    else:
        assert expected == actual

    return True


def flatten_obj(obj: Any) -> List[tuple[str, Any]]:
    acc = []

    def rec(path: str, x: Any):
        nonlocal acc

        if isinstance(x, dict):
            for key in sorted(x.keys()):
                rec(path + "." + key, x[key])
        elif isinstance(x, (list, tuple)):
            for i in range(len(x)):
                rec(path + "." + str(i), x[i])
        else:
            acc.append((path.lstrip("."), x))

    rec("", obj)

    return acc
