import asyncio
from typing import Any

import pytest

from aidial_adapter_vertexai.utils.cache import cache


async def test_concurrent_calls_same_key_are_single_flight():
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    @cache()
    async def f(key: str) -> int:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return 10

    tasks = [asyncio.create_task(f("x")) for _ in range(10)]

    await entered.wait()

    assert calls == 1

    release.set()
    results = await asyncio.gather(*tasks)

    assert results == [10] * 10
    assert calls == 1


async def test_sequential_calls_same_key_use_cached_value():
    calls = 0

    @cache()
    async def f(key: str) -> int:
        nonlocal calls
        calls += 1
        return 42

    assert await f("x") == 42
    assert await f("x") == 42
    assert await f("x") == 42
    assert calls == 1


async def test_different_keys_are_computed_independently():
    calls: list[str] = []

    @cache()
    async def f(key: str) -> str:
        calls.append(key)
        await asyncio.sleep(0.01)
        return f"value:{key}"

    results = await asyncio.gather(
        f("a"),
        f("b"),
        f("a"),
        f("c"),
        f("b"),
    )

    assert results == [
        "value:a",
        "value:b",
        "value:a",
        "value:c",
        "value:b",
    ]
    assert calls.count("a") == 1
    assert calls.count("b") == 1
    assert calls.count("c") == 1


async def test_kwargs_order_does_not_change_cache_key():
    calls = 0

    @cache()
    async def f(*, a: int, b: int) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert await f(a=1, b=2) == 3
    assert await f(b=2, a=1) == 3
    assert calls == 1


async def test_args_are_part_of_cache_key():
    calls = 0

    @cache()
    async def f(a: int, b: int) -> int:
        nonlocal calls
        calls += 1
        return 10 * a + b

    assert await f(1, 2) == 12
    assert await f(2, 1) == 21
    assert calls == 2


async def test_exception_is_not_cached():
    calls = 0

    @cache()
    async def f(key: str) -> int:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"boom:{key}:{calls}")

    with pytest.raises(RuntimeError, match=r"boom:x:1"):
        await f("x")

    with pytest.raises(RuntimeError, match=r"boom:x:2"):
        await f("x")

    assert calls == 2


async def test_concurrent_waiters_for_same_failing_key_see_same_exception_instance():
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    @cache()
    async def f(key: str) -> int:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise ValueError("boom")

    tasks = [asyncio.create_task(f("x")) for _ in range(5)]

    await entered.wait()
    assert calls == 1

    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(x, ValueError) for x in results)
    assert all(str(x) == "boom" for x in results)
    assert calls == 1


async def test_clear_removes_cached_entries():
    calls = 0

    @cache()
    async def f(key: str) -> str:
        nonlocal calls
        calls += 1
        return f"v:{key}:{calls}"

    assert await f("x") == "v:x:1"
    assert await f("x") == "v:x:1"
    assert calls == 1

    await f.clear()

    assert await f("x") == "v:x:2"
    assert calls == 2


async def test_clear_calls_close_for_resolved_values():
    closed: list[str] = []

    async def close(value: str):
        closed.append(value)

    @cache(close=close)
    async def f(key: str) -> str:
        return f"value:{key}"

    assert await f("a") == "value:a"
    assert await f("b") == "value:b"
    assert await f("a") == "value:a"

    await f.clear()

    assert sorted(closed) == ["value:a", "value:b"]


async def test_clear_on_empty_cache_is_noop():
    closed: list[Any] = []

    async def close(value: Any):
        closed.append(value)

    @cache(close=close)
    async def f(key: str) -> str:
        return key

    await f.clear()
    assert closed == []


async def test_clear_does_not_close_failed_entries():
    closed: list[str] = []

    async def close(value: str):
        closed.append(value)

    @cache(close=close)
    async def f(key: str) -> str:
        if key == "bad":
            raise RuntimeError("boom")
        return f"value:{key}"

    assert await f("ok") == "value:ok"
    with pytest.raises(RuntimeError, match="boom"):
        await f("bad")

    await f.clear()

    assert closed == ["value:ok"]


async def test_clear_does_not_close_pending_entries():
    closed: list[str] = []
    release = asyncio.Event()

    async def close(value: str):
        closed.append(value)

    @cache(close=close)
    async def f(key: str) -> str:
        await release.wait()
        return f"value:{key}"

    task = asyncio.create_task(f("x"))

    await f.clear()
    assert closed == []

    release.set()
    assert await task == "value:x"


async def test_different_decorated_functions_have_independent_caches():
    calls_f = 0
    calls_g = 0

    @cache()
    async def f(key: str) -> str:
        nonlocal calls_f
        calls_f += 1
        return f"f:{key}"

    @cache()
    async def g(key: str) -> str:
        nonlocal calls_g
        calls_g += 1
        return f"g:{key}"

    assert await f("x") == "f:x"
    assert await f("x") == "f:x"
    assert await g("x") == "g:x"
    assert await g("x") == "g:x"

    assert calls_f == 1
    assert calls_g == 1


async def test_same_key_after_clear_recomputes_once_under_concurrency() -> None:
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    @cache()
    async def f(key: str) -> int:
        nonlocal calls, entered, release
        calls += 1
        entered.set()
        await release.wait()
        return calls

    for gen in [1, 2]:
        await f.clear()

        entered = asyncio.Event()
        release = asyncio.Event()

        tasks = [asyncio.create_task(f("x")) for _ in range(5)]

        await entered.wait()
        assert calls == gen

        release.set()
        results = await asyncio.gather(*tasks)

        assert results == [gen] * 5
        assert calls == gen
