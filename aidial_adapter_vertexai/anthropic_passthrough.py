from fastapi import Request

from aidial_adapter_vertexai.upstream_config import (
    AnthropicClient,
    parse_upstream_config,
)


async def get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = parse_upstream_config(request)
    return await upstream_config.get_anthropic_client()
