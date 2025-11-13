import os

import anthropic
import httpx
import vertexai
from anthropic import AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient

from aidial_adapter_vertexai.utils.cache import cache
from aidial_adapter_vertexai.utils.log_config import app_logger as log

DEFAULT_REGION_ENV_VAR = "DEFAULT_REGION"
DEFAULT_PROJECT_ENV_VAR = "GCP_PROJECT_ID"

DEFAULT_REGION = os.getenv(DEFAULT_REGION_ENV_VAR)
DEFAULT_PROJECT = os.getenv(DEFAULT_PROJECT_ENV_VAR)


def init_vertex_ai():
    if DEFAULT_REGION and DEFAULT_PROJECT:
        vertexai.init(project=DEFAULT_PROJECT, location=DEFAULT_REGION)
    else:
        log.warning(
            f"{DEFAULT_REGION_ENV_VAR!r} and {DEFAULT_PROJECT_ENV_VAR!r} aren't configured."
        )


async def _close_genai_client(client: GenAIClient) -> None:
    if session := client._api_client._aiohttp_session:
        await session.close()


@cache(_close_genai_client)
async def get_genai_client(project: str, location: str) -> GenAIClient:
    return GenAIClient(vertexai=True, project=project, location=location)


async def _close_anthropic_client(client: AsyncAnthropicVertex) -> None:
    await client.close()


@cache(_close_anthropic_client)
async def get_anthropic_client(
    project: str, region: str
) -> AsyncAnthropicVertex:
    http_client = httpx.AsyncClient(timeout=_get_default_anthropic_timeout())
    return AsyncAnthropicVertex(
        project_id=project, region=region, http_client=http_client
    )


def _get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)
