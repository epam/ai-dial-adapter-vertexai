from aidial_adapter_anthropic.passthrough import mount_anthropic_api
from fastapi import FastAPI, Request

from aidial_adapter_vertexai.upstream_config import (
    AnthropicClient,
    parse_upstream_config,
)


async def _get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = parse_upstream_config(request)
    return await upstream_config.get_anthropic_client()


def mount_anthropic_passthrough(app: FastAPI, path: str):
    mount_anthropic_api(app, _get_anthropic_client, path=path)
