from functools import lru_cache

import vertexai
from anthropic import AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient

from aidial_adapter_vertexai.utils.env import get_env

LOCATION = get_env("DEFAULT_REGION")
PROJECT_ID = get_env("GCP_PROJECT_ID")


def init_vertex_ai():
    vertexai.init(project=PROJECT_ID, location=LOCATION)


@lru_cache(None)
def get_genai_client(location: str) -> GenAIClient:
    return GenAIClient(vertexai=True, project=PROJECT_ID, location=location)


@lru_cache(None)
def get_anthropic_client(location: str) -> AsyncAnthropicVertex:
    return AsyncAnthropicVertex(project_id=PROJECT_ID, region=location)
