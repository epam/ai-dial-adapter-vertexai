import os

import anthropic
import httpx
import vertexai
from anthropic import AsyncAnthropicFoundry, AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions, HttpRetryOptions

from aidial_adapter_vertexai.aws_credentials import maybe_make_aws_credentials
from aidial_adapter_vertexai.utils.azure_auth import get_azure_access_token
from aidial_adapter_vertexai.utils.cache import cache
from aidial_adapter_vertexai.utils.env import get_env_int
from aidial_adapter_vertexai.utils.log_config import app_logger as log

DEFAULT_REGION_ENV_VAR = "DEFAULT_REGION"
DEFAULT_PROJECT_ENV_VAR = "GCP_PROJECT_ID"


def get_default_region() -> str | None:
    return os.getenv(DEFAULT_REGION_ENV_VAR)


def get_default_project() -> str | None:
    return os.getenv(DEFAULT_PROJECT_ENV_VAR)


ANTHROPIC_MAX_RETRY_ATTEMPTS = get_env_int("ANTHROPIC_MAX_RETRY_ATTEMPTS", 0)
GOOGLE_GENAI_MAX_RETRY_ATTEMPTS = get_env_int(
    "GOOGLE_GENAI_MAX_RETRY_ATTEMPTS", 0
)


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
        retry_options=HttpRetryOptions(
            attempts=1 + GOOGLE_GENAI_MAX_RETRY_ATTEMPTS
        )
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
    http_client = httpx.AsyncClient(timeout=_get_default_anthropic_timeout())
    creds = maybe_make_aws_credentials()
    return AsyncAnthropicVertex(
        project_id=project,
        region=region,
        http_client=http_client,
        max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
        credentials=creds,
    )


async def _close_anthropic_httpx_client(client: httpx.AsyncClient) -> None:
    await client.aclose()


@cache(_close_anthropic_httpx_client)
async def get_anthropic_httpx_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_get_default_anthropic_timeout())


async def get_anthropic_foundry_client(
    api_key: str | None, base_url: str
) -> AsyncAnthropicFoundry:
    token_provider = get_azure_access_token if api_key is None else None
    return AsyncAnthropicFoundry(
        api_key=api_key,
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        http_client=await get_anthropic_httpx_client(base_url),
        max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
    )


def _get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)
