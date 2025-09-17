import dataclasses
from typing import List, Mapping, Tuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.adapter_deployments import (
    AdapterChatCompletionDeployment,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from tests.utils.openai import sanitize_test_name, sys, user

_CENTRAL = "us-central1"
_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.GEMINI_2_5_PRO: _CENTRAL,
    D.GEMINI_2_5_FLASH: _CENTRAL,
}

deployments = list(_DEPLOYMENT_TO_REGION.keys())


@pytest.fixture
def deployment(request) -> D:
    return request.param


def display_deployment(dep: D):
    return sanitize_test_name(dep.value)


@pytest.fixture
def region(deployment: D) -> str:
    region = _DEPLOYMENT_TO_REGION.get(deployment)
    if region is None:
        raise ValueError(
            f"{deployment.value!r} is missing from the region mapping"
        )
    return region


MockPrompt = Tuple[str, dict | None]


class MockChatCompletionAdapter(ChatCompletionAdapter[MockPrompt]):
    deployment: AdapterChatCompletionDeployment

    def __init__(self, deployment: AdapterChatCompletionDeployment) -> None:
        self.deployment = deployment

    async def parse_prompt(
        self,
        params: ModelParameters,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> MockPrompt | UserError:
        return str(messages[-1].content), params.configuration

    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: MockPrompt
    ) -> None:
        await consumer.append_content("hello world")

        if usage := prompt[1]:
            await consumer.set_usage(
                TokenUsage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                )
            )

    async def count_prompt_tokens(self, prompt: MockPrompt) -> int:
        return len(prompt[0].split())


@pytest.fixture(autouse=True)
def mock_adapter():
    async def _mock_adapter(
        *,
        api_key: str,
        upstream_config: UpstreamConfig,
        deployment: AdapterChatCompletionDeployment,
    ):
        return MockChatCompletionAdapter(deployment=deployment)

    with patch(
        "aidial_adapter_vertexai.adapters.get_chat_completion_model",
        new=AsyncMock(side_effect=_mock_adapter),
    ):
        yield


token_threshold = 4_096
big_content = "cat " * 10_000  # #tokens >= token_threshold
small_content = "cat"  # #tokens < token_threshold

big_usage = {
    "prompt_tokens": token_threshold,
    "completion_tokens": 1,
    "total_tokens": token_threshold + 1,
}

small_usage = {
    "prompt_tokens": big_usage["prompt_tokens"] - 1,
    "completion_tokens": 1,
    "total_tokens": big_usage["total_tokens"] - 1,
}


@dataclasses.dataclass
class DialCacheTestCase:
    __test__ = False

    stream: bool
    request_content: str
    request_usage: dict | None
    caching_enabled: bool

    expected_caching_response: bool

    def get_name(self):
        xs = []
        if self.stream:
            xs.append("stream")
        else:
            xs.append("block")

        if len(self.request_content) >= token_threshold:
            xs.append("big-content")
        else:
            xs.append("small-content")

        if self.caching_enabled:
            xs.append("caching")
        else:
            xs.append("no-caching")

        if self.request_usage:
            if self.request_usage["prompt_tokens"] >= token_threshold:
                xs.append("big-usage")
            else:
                xs.append("small-usage")
        else:
            xs.append("no-usage")

        return "/".join(xs)


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
@pytest.mark.parametrize(
    "ts",
    [
        ts
        for stream in [True, False]
        for ts in [
            DialCacheTestCase(stream, big_content, None, True, True),
            DialCacheTestCase(stream, big_content, None, False, False),
            DialCacheTestCase(stream, small_content, None, True, False),
            DialCacheTestCase(stream, small_content, None, False, False),
            DialCacheTestCase(stream, big_content, small_usage, True, stream),
            DialCacheTestCase(stream, big_content, small_usage, False, False),
            DialCacheTestCase(
                stream, small_content, big_usage, True, not stream
            ),
            DialCacheTestCase(stream, small_content, big_usage, False, False),
        ]
    ],
    ids=lambda x: x.get_name(),
)
async def test_dial_cache(
    test_http_client: httpx.AsyncClient, deployment: D, ts: DialCacheTestCase
):
    headers = {}
    if ts.caching_enabled:
        headers["X-DIAL-CACHE-BREAKPOINT-PATH"] = "whatever"

    response = await test_http_client.post(
        url=f"/openai/deployments/{deployment.value}/chat/completions",
        json={
            "messages": [sys("be helpful"), user(ts.request_content)],
            "custom_fields": {"configuration": ts.request_usage},
            "stream": ts.stream,
        },
        headers=headers,
    )

    assert response.status_code == 200

    cache_path = response.headers.get("X-DIAL-CACHE-BREAKPOINT-PATH")
    expire_at = response.headers.get("X-DIAL-CACHE-EXPIRE-AT")

    if ts.expected_caching_response:
        assert cache_path == "prefix.body.messages[1]"
        assert expire_at is not None
    else:
        assert cache_path is None
        assert expire_at is None
