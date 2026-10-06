import anthropic
import httpx
from anthropic import AsyncAnthropic
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions
from mistralai.client import Mistral

import aidial_adapter_vertexai.upstream_config as upstream_config_module
from aidial_adapter_vertexai.clients import get_httpx_client
from aidial_adapter_vertexai.dial_api.exceptions import to_dial_exception
from aidial_adapter_vertexai.upstream_config import (
    _ApiKeyUpstreamConfig,
    parse_upstream_config,
)
from aidial_adapter_vertexai.utils.constants import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_POOL_TIMEOUT,
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

    monkeypatch.setenv("DEFAULT_REGION", "global")
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
    pool = client._transport._pool  # type: ignore[attr-defined]
    assert pool._max_connections == HTTP_MAX_CONNECTIONS
    assert pool._max_keepalive_connections == HTTP_MAX_KEEPALIVE_CONNECTIONS


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
