import os
from functools import cache

import vertexai
from anthropic import AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient

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


@cache
def get_genai_client(project: str, location: str) -> GenAIClient:
    return GenAIClient(vertexai=True, project=project, location=location)


@cache
def get_anthropic_client(project: str, region: str) -> AsyncAnthropicVertex:
    return AsyncAnthropicVertex(project_id=project, region=region)
