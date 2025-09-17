import dataclasses
from typing import Tuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from aidial_adapter_vertexai.adapter_deployments import (
    AdapterChatCompletionDeployment,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.dial_api.caching import get_prompt_tokens_threshold
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from tests.utils.openai import sanitize_test_name, sys, user

_DEPLOYMENTS = [
    D.GEMINI_2_5_PRO,
    D.GEMINI_2_5_FLASH,
    D.CLAUDE_3_5_HAIKU,  # The deployment that doesn't support implicit caching
]


_MockPrompt = Tuple[str, dict | None]


class _MockChatCompletionAdapter(ChatCompletionAdapter[_MockPrompt]):
    deployment: AdapterChatCompletionDeployment

    def __init__(self, deployment: AdapterChatCompletionDeployment) -> None:
        self.deployment = deployment

    async def parse_prompt(
        self, params: ModelParameters, tools, static_tools, messages
    ) -> _MockPrompt | UserError:
        return str(messages[-1].content), params.configuration

    async def chat(
        self, params, consumer: Consumer, prompt: _MockPrompt
    ) -> None:
        await consumer.append_content("hello world")

        if usage := prompt[1]:
            await consumer.set_usage(
                TokenUsage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                )
            )

    async def count_prompt_tokens(self, prompt: _MockPrompt) -> int:
        return len(prompt[0].split())


@pytest.fixture(autouse=True)
def mock_adapter():
    async def _mock_adapter(
        *, api_key, upstream_config, deployment: AdapterChatCompletionDeployment
    ):
        return _MockChatCompletionAdapter(deployment=deployment)

    with patch(
        "aidial_adapter_vertexai.chat_completion.get_chat_completion_model",
        new=AsyncMock(side_effect=_mock_adapter),
    ):
        yield


@dataclasses.dataclass
class DialCacheTestCase:
    __test__ = False

    deployment: D
    stream: bool
    is_big_content: bool
    is_big_usage: bool | None
    caching_enabled: bool

    expected_caching_headers: bool

    @property
    def are_caching_headers_expected(self) -> bool:
        if get_prompt_tokens_threshold(self.deployment) is None:
            return False
        return self.expected_caching_headers

    @property
    def request_content(self) -> str:
        if self.is_big_content:
            return "cat " * (self.token_threshold + 1)
        else:
            return "cat"

    @property
    def request_usage(self) -> dict | None:
        if self.is_big_usage is None:
            return None
        if self.is_big_usage:
            return {
                "prompt_tokens": self.token_threshold,
                "completion_tokens": 1,
                "total_tokens": self.token_threshold + 1,
            }
        else:
            return {
                "prompt_tokens": self.token_threshold - 1,
                "completion_tokens": 1,
                "total_tokens": self.token_threshold,
            }

    @property
    def token_threshold(self) -> int:
        return get_prompt_tokens_threshold(self.deployment) or 10

    def get_name(self):
        xs = []

        xs.append(self.deployment.value)

        if self.stream:
            xs.append("stream")
        else:
            xs.append("block")

        if self.is_big_content:
            xs.append("big-content")
        else:
            xs.append("small-content")

        if self.caching_enabled:
            xs.append("caching")
        else:
            xs.append("no-caching")

        if self.is_big_usage is not None:
            if self.is_big_usage:
                xs.append("big-usage")
            else:
                xs.append("small-usage")
        else:
            xs.append("no-usage")

        return sanitize_test_name("/".join(xs))


@pytest.mark.parametrize(
    "ts",
    [
        ts
        for stream in [True, False]
        for deployment in _DEPLOYMENTS
        for ts in [
            DialCacheTestCase(deployment, stream, True, None, True, True),
            DialCacheTestCase(deployment, stream, True, None, False, False),
            DialCacheTestCase(deployment, stream, False, None, True, False),
            DialCacheTestCase(deployment, stream, False, None, False, False),
            DialCacheTestCase(deployment, stream, True, False, True, stream),
            DialCacheTestCase(deployment, stream, True, False, False, False),
            DialCacheTestCase(
                deployment, stream, False, True, True, not stream
            ),
            DialCacheTestCase(deployment, stream, False, True, False, False),
        ]
    ],
    ids=lambda x: x.get_name(),
)
async def test_dial_cache(
    test_http_client: httpx.AsyncClient, ts: DialCacheTestCase
):
    headers = {}
    if ts.caching_enabled:
        headers["X-DIAL-CACHE-BREAKPOINT-PATH"] = "whatever"

    response = await test_http_client.post(
        url=f"/openai/deployments/{ts.deployment.value}/chat/completions",
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

    if ts.are_caching_headers_expected:
        assert cache_path == "prefix.body.messages[1]"
        assert expire_at is not None
    else:
        assert cache_path is None
        assert expire_at is None
