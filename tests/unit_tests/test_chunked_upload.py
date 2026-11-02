import os

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aidial_adapter_vertexai.clients import (
    _UPLOAD_CHUNK_SIZE,
    _AiohttpTransport,
    _SharedAiohttpClient,
    _SharedAsyncClient,
    _with_chunked_body,
)

_LARGE_BODY = os.urandom(3 * _UPLOAD_CHUNK_SIZE + 123)
_SMALL_BODY = b'{"contents": []}'


def _aiohttp_client() -> httpx.AsyncClient:
    return _SharedAiohttpClient(transport=_AiohttpTransport())


def _httpx_client() -> httpx.AsyncClient:
    return _SharedAsyncClient()


@pytest.fixture
async def echo_server():
    received: list[web.Request] = []
    bodies: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        received.append(request)
        bodies.append(await request.read())
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=16 * _UPLOAD_CHUNK_SIZE)
    app.router.add_post("/", handler)
    async with TestServer(app) as server:
        yield server, received, bodies


@pytest.mark.parametrize(
    "body", [_LARGE_BODY, _SMALL_BODY], ids=["large", "small"]
)
@pytest.mark.parametrize(
    "make_client", [_aiohttp_client, _httpx_client], ids=["aiohttp", "httpx"]
)
async def test_upload_through_transport(echo_server, make_client, body: bytes):
    server, received, bodies = echo_server

    async with make_client() as client:
        response = await client.post(
            str(server.make_url("/")),
            content=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}

    [request] = received
    assert bodies == [body]
    assert request.headers["Content-Length"] == str(len(body))
    assert "Transfer-Encoding" not in request.headers


async def test_large_body_is_split_into_views():
    request = httpx.Request("POST", "https://upstream", content=_LARGE_BODY)

    chunked = _with_chunked_body(request)
    chunks = [chunk async for chunk in chunked.stream]  # type: ignore

    assert [len(c) for c in chunks] == [_UPLOAD_CHUNK_SIZE] * 3 + [123]
    assert all(
        isinstance(c, memoryview) and c.obj is _LARGE_BODY for c in chunks
    )
    assert chunked.headers["Content-Length"] == str(len(_LARGE_BODY))

    # re-iterable, e.g. for redirects
    assert b"".join([bytes(c) async for c in chunked.stream]) == _LARGE_BODY  # type: ignore


def test_small_body_is_sent_as_is():
    request = httpx.Request("POST", "https://upstream", content=_SMALL_BODY)
    assert _with_chunked_body(request) is request
