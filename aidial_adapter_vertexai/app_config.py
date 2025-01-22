import vertexai
from anthropic import AsyncAnthropicVertex
from google.genai.client import Client as GenAIClient

from aidial_adapter_vertexai.utils.env import get_env

LOCATION = get_env("DEFAULT_REGION")
PROJECT_ID = get_env("GCP_PROJECT_ID")


def init_vertex_ai():
    vertexai.init(project=PROJECT_ID, location=LOCATION)


GENAI_CLIENT = GenAIClient(vertexai=True, project=PROJECT_ID, location=LOCATION)

ANTHROPIC_CLIENT = AsyncAnthropicVertex(project_id=PROJECT_ID, region=LOCATION)
