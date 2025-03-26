from functools import cache

import vertexai
from anthropic import AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient

from aidial_adapter_vertexai.utils.env import get_env

DEFAULT_REGION = get_env("DEFAULT_REGION")
DEFAULT_PROJECT = get_env("GCP_PROJECT_ID")


def init_vertex_ai():
    vertexai.init(project=DEFAULT_PROJECT, location=DEFAULT_REGION)


@cache
def get_genai_client(project: str, location: str) -> GenAIClient:
    return GenAIClient(vertexai=True, project=project, location=location)


@cache
def get_anthropic_client(project: str, region: str) -> AsyncAnthropicVertex:
    return AsyncAnthropicVertex(project_id=project, region=region)
