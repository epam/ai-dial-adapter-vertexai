import os

import anthropic
import httpx
import vertexai
from anthropic import AsyncAnthropicFoundry, AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions, HttpRetryOptions
from mistralai.gcp.client import MistralGCP

from aidial_adapter_vertexai.aws_credentials import maybe_make_aws_credentials
from aidial_adapter_vertexai.utils.azure_auth import get_azure_access_token
from aidial_adapter_vertexai.utils.cache import cache
from aidial_adapter_vertexai.utils.constants import (
    ANTHROPIC_MAX_RETRY_ATTEMPTS,
    DEFAULT_PROJECT_ENV_VAR,
    DEFAULT_REGION_ENV_VAR,
    GOOGLE_GENAI_MAX_RETRY_ATTEMPTS,
    HTTP_CONNECT_TIMEOUT,
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_POOL_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
)
from aidial_adapter_vertexai.utils.log_config import app_logger as log


def get_default_region() -> str | None:
    return os.getenv(DEFAULT_REGION_ENV_VAR)


def get_default_project() -> str | None:
    return os.getenv(DEFAULT_PROJECT_ENV_VAR)


def init_vertex_ai():
    if (region := get_default_region()) and (project := get_default_project()):
        creds = maybe_make_aws_credentials()
        vertexai.init(project=project, location=region, credentials=creds)
    else:
        log.warning(
            f"{DEFAULT_REGION_ENV_VAR!r} and {DEFAULT_PROJECT_ENV_VAR!r} aren't configured."
        )


async def _close_genai_client(client: GenAIClient) -> None:
    await client.aio.aclose()


@cache(_close_genai_client)
async def get_genai_client(project: str, location: str) -> GenAIClient:
    opts = HttpOptions(
        httpx_async_client=await get_httpx_client(),
        retry_options=HttpRetryOptions(
            attempts=1 + GOOGLE_GENAI_MAX_RETRY_ATTEMPTS
        ),
    )
    creds = maybe_make_aws_credentials()
    return GenAIClient(
        vertexai=True,
        project=project,
        location=location,
        credentials=creds,
        http_options=opts,
    )


async def _close_anthropic_client(client: AsyncAnthropicVertex) -> None:
    await client.close()


@cache(_close_anthropic_client)
async def get_anthropic_vertex_client(
    project: str, region: str
) -> AsyncAnthropicVertex:
    creds = maybe_make_aws_credentials()
    return AsyncAnthropicVertex(
        project_id=project,
        region=region,
        http_client=await get_httpx_client(),
        max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
        credentials=creds,
    )


class _SharedAsyncClient(httpx.AsyncClient):
    def build_request(
        self, *args, timeout=httpx.USE_CLIENT_DEFAULT, **kwargs
    ) -> httpx.Request:
        # Google GenAI SDK explicitly passes `timeout=None` to the httpx client
        # whenever `HttpOptions.timeout` isn't configured. Httpx treats it as
        # "no timeouts at all", which disables the pool timeout as well.
        # Fall back to the client-wide timeouts instead.
        if timeout is None:
            timeout = httpx.USE_CLIENT_DEFAULT
        return super().build_request(*args, timeout=timeout, **kwargs)


async def _close_httpx_client(client: httpx.AsyncClient) -> None:
    await client.aclose()


@cache(_close_httpx_client)
async def get_httpx_client() -> httpx.AsyncClient:
    return _SharedAsyncClient(
        follow_redirects=True,
        timeout=_get_http_timeouts(),
        limits=_get_http_limits(),
    )


async def get_anthropic_foundry_client(
    api_key: str | None, base_url: str
) -> AsyncAnthropicFoundry:
    token_provider = get_azure_access_token if api_key is None else None
    return AsyncAnthropicFoundry(
        api_key=api_key,
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        http_client=await get_httpx_client(),
        max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
    )


async def _close_mistral_gcp_client(client: MistralGCP):
    if client.sdk_configuration.async_client:
        await client.sdk_configuration.async_client.aclose()


@cache(_close_mistral_gcp_client)
async def get_mistral_gcp_client(project_id: str, region: str) -> MistralGCP:
    return MistralGCP(
        project_id=project_id,
        region=region,
        async_client=await get_httpx_client(),
    )


def _get_http_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
    )


def _get_http_timeouts() -> httpx.Timeout:
    timeout = httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        pool=HTTP_POOL_TIMEOUT,
    )

    if timeout == anthropic._constants.DEFAULT_TIMEOUT:
        # Providing a timeout marginally different from the default Anthropic timeout
        # in order to disable the check that throws an error when
        # stream=False & max_tokens>=128K/6:
        # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176
        timeout_dict = timeout.as_dict()
        timeout_dict["connect"] *= 1.0001  # type: ignore
        timeout = httpx.Timeout(**timeout_dict)

    return timeout
