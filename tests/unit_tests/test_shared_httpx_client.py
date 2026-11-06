import gc
import weakref

import httpx
import pytest

from aidial_adapter_vertexai.utils.httpx import (
    SharedAiohttpClient,
    SharedAsyncClient,
)


def _transport() -> httpx.MockTransport:
    # A stream (unlike content) isn't read upfront, like in the real transports
    return httpx.MockTransport(
        lambda _: httpx.Response(200, stream=httpx.ByteStream(b"ok"))
    )


@pytest.mark.parametrize("stream", [False, True], ids=["block", "stream"])
@pytest.mark.parametrize("client_cls", [SharedAsyncClient, SharedAiohttpClient])
async def test_request_is_released_without_gc(
    client_cls: type[httpx.AsyncClient], stream: bool
):
    async with client_cls(transport=_transport()) as client:
        gc.disable()
        try:
            request = client.build_request(
                "POST", "https://upstream", content=b"payload"
            )
            request_ref = weakref.ref(request)

            response = await client.send(request, stream=stream)
            if stream:
                assert await response.aread() == b"ok"
            else:
                assert response.content == b"ok"
            assert response.is_closed

            del request, response
            is_released = request_ref() is None
        finally:
            gc.enable()

    assert is_released
