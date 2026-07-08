from typing import assert_never

from aidial_adapter_anthropic.passthrough import mount_anthropic_api
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from fastapi import FastAPI, Request

from aidial_adapter_vertexai.upstream_config import (
    AnthropicClient,
    parse_upstream_config,
)


async def _get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = parse_upstream_config(request)
    return await upstream_config.get_anthropic_client()


def _strip_unsupported_features(
    client: AnthropicClient, features: list[str]
) -> list[str]:
    _unsupported_flags_by_vertex = {
        "thinking-token-count-2026-05-13",
        "prompt-caching-scope-2026-01-05",
        "advisor-tool-2026-03-01",
    }
    _unsupported_flags_by_azure = {"advisor-tool-2026-03-01"}
    match client:
        case AsyncAnthropicFoundry():
            return [f for f in features if f not in _unsupported_flags_by_azure]
        case AsyncAnthropicVertex():
            return [
                f for f in features if f not in _unsupported_flags_by_vertex
            ]
        case AsyncAnthropic():
            return features
        case _:
            assert_never(client)


def mount_anthropic_passthrough(app: FastAPI, path: str):
    mount_anthropic_api(
        app,
        _get_anthropic_client,
        path=path,
        on_anthropic_beta_header=_strip_unsupported_features,
    )
