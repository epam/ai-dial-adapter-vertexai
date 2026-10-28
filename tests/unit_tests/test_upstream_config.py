import asyncio

import anthropic
import httpx
import pytest
from aiohttp import web
from anthropic import AsyncAnthropic
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions
from httpx_aiohttp import AiohttpTransport
from mistralai.client import Mistral

import aidial_adapter_vertexai.upstream_config as upstream_config_module
from aidial_adapter_vertexai.chat.gemini.generation_config import (
    to_genai_http_options,
)
from aidial_adapter_vertexai.clients import _AiohttpTransport, get_httpx_client
from aidial_adapter_vertexai.dial_api.exceptions import to_dial_exception
from aidial_adapter_vertexai.upstream_config import (
    _ApiKeyUpstreamConfig,
    parse_upstream_config,
)
from aidial_adapter_vertexai.utils.constants import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_POOL_TIMEOUT,
    HTTP_TRANSPORT,
)


class _RequestStub:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def _fake_get_genai_client(called: dict[str, str]):
    async def _wrapped(project: str, location: str):
        called["project"] = project
        called["location"] = location
        return object()

    return _wrapped


async def test_parse_upstream_config_returns_mistral_client_for_api_key():
    request = _RequestStub(headers={"x-upstream-key": "test-key"})

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    client = await config.get_mistral_client()

    assert isinstance(client, Mistral)


async def test_parse_upstream_config_uses_region_and_project_from_header(
    monkeypatch,
):
    called: dict[str, str] = {}

    monkeypatch.setattr(
        upstream_config_module,
        "get_genai_client",
        _fake_get_genai_client(called),
    )
    request = _RequestStub(
        headers={
            "x-upstream-extra-data": '{"region":"eu","project":"my-project"}'
        }
    )

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    await config.get_genai_client()

    assert called == {"project": "my-project", "location": "eu"}


async def test_parse_upstream_config_falls_back_to_default_env(
    monkeypatch,
):
    called: dict[str, str] = {}

    monkeypatch.setenv("GCP_DEFAULT_REGION", "global")
    monkeypatch.setenv("GCP_PROJECT_ID", "project_id")
    monkeypatch.setattr(
        upstream_config_module,
        "get_genai_client",
        _fake_get_genai_client(called),
    )
    request = _RequestStub(headers={})

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    await config.get_genai_client()

    assert called == {"project": "project_id", "location": "global"}


async def test_api_key_anthropic_client_uses_shared_httpx_client():
    config = _ApiKeyUpstreamConfig(api_key="test-key")
    client: AsyncAnthropic = await config.get_anthropic_client()
    assert client._client is await get_httpx_client()


async def test_api_key_genai_client_uses_shared_httpx_client():
    config = _ApiKeyUpstreamConfig(api_key="test-key")
    client: GenAIClient = await config.get_genai_client()
    assert client._api_client._async_httpx_client is await get_httpx_client()


async def test_api_key_mistral_client_uses_shared_httpx_client():
    config = _ApiKeyUpstreamConfig(api_key="test-key")
    client: Mistral = await config.get_mistral_client()
    assert client.sdk_configuration.async_client is await get_httpx_client()


async def test_genai_client_doesnt_disable_shared_httpx_timeouts():
    # The GenAI SDK explicitly passes `timeout=None` to the httpx client,
    # which would otherwise disable all the timeouts of the shared client.
    timeouts: list[dict | None] = []

    class _SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            timeouts.append(request.extensions.get("timeout"))
            return httpx.Response(200, json={"candidates": []})

    shared_client = await get_httpx_client()
    client = GenAIClient(
        api_key="test-key",
        http_options=HttpOptions(
            httpx_async_client=type(shared_client)(
                transport=_SpyTransport(), timeout=shared_client.timeout
            )
        ),
    )

    await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents="hello"
    )

    assert timeouts == [shared_client.timeout.as_dict()]


async def test_shared_httpx_client_uses_configured_connection_limits():
    client = await get_httpx_client()
    transport = client._transport  # type: ignore[attr-defined]
    if HTTP_TRANSPORT == "aiohttp":
        assert isinstance(transport, _AiohttpTransport)
        assert transport.limits.max_connections == HTTP_MAX_CONNECTIONS
    else:
        pool = transport._pool  # type: ignore[attr-defined]
        assert pool._max_connections == HTTP_MAX_CONNECTIONS
        assert pool._max_keepalive_connections == HTTP_MAX_KEEPALIVE_CONNECTIONS


async def test_aiohttp_pool_exhaustion_is_pool_timeout(unused_tcp_port: int):
    release = asyncio.Event()

    async def slow(_: web.Request) -> web.Response:
        await release.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", slow)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()

    client = httpx.AsyncClient(
        transport=_AiohttpTransport(limits=httpx.Limits(max_connections=1)),
        timeout=httpx.Timeout(5.0, pool=0.2),
    )
    url = f"http://127.0.0.1:{unused_tcp_port}/"
    holder = asyncio.create_task(client.get(url))
    try:
        await asyncio.sleep(0.1)  # the only connection is now in use

        with pytest.raises(httpx.PoolTimeout) as exc_info:
            await client.get(url)
        assert to_dial_exception(exc_info.value).status_code == 503
    finally:
        release.set()
        assert (await holder).status_code == 200
        await client.aclose()
        await runner.cleanup()


async def test_aiohttp_connect_timeout_without_queueing_stays_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _connect_timeout(self, request: httpx.Request):
        raise httpx.ConnectTimeout("slow connect", request=request)

    monkeypatch.setattr(
        AiohttpTransport, "handle_async_request", _connect_timeout
    )
    transport = _AiohttpTransport()
    with pytest.raises(httpx.ConnectTimeout) as exc_info:
        await transport.handle_async_request(
            httpx.Request("GET", "http://example.com")
        )
    assert not isinstance(exc_info.value, httpx.PoolTimeout)


async def test_shared_httpx_client_uses_configured_pool_timeout():
    client = await get_httpx_client()
    assert client.timeout.pool == HTTP_POOL_TIMEOUT


def test_pool_timeout_maps_to_503():
    request = httpx.Request("POST", "https://example.com")
    pool_timeout = httpx.PoolTimeout("pool exhausted", request=request)
    try:
        raise anthropic.APITimeoutError(request=request) from pool_timeout
    except anthropic.APITimeoutError as wrapped:
        anthropic_error = wrapped

    for e in [pool_timeout, anthropic_error]:
        assert to_dial_exception(e).status_code == 503


async def test_vertex_ai_headers_are_passed_on_to_genai():
    config = parse_upstream_config(
        _RequestStub(  # type: ignore[arg-type]
            headers={
                "x-upstream-extra-data": '{"region":"eu","project":"p"}',
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "authorization": "secret",
            }
        )
    )

    sent: list[httpx.Headers] = []

    class _SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            sent.append(request.headers)
            return httpx.Response(200, json={"candidates": []})

    client = GenAIClient(
        api_key="test-key",
        http_options=HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=_SpyTransport())
        ),
    )

    await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="hello",
        config={"http_options": to_genai_http_options(config.get_headers())},
    )

    assert sent[0]["x-vertex-ai-llm-request-type"] == "shared"
    assert "authorization" not in sent[0]


async def test_vertex_ai_headers_arent_passed_on_to_other_upstreams():
    config = parse_upstream_config(
        _RequestStub(  # type: ignore[arg-type]
            headers={
                "x-upstream-key": "test-key",
                "X-Vertex-AI-LLM-Request-Type": "shared",
            }
        )
    )

    assert config.get_headers() == {}
