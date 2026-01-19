import dataclasses
from typing import Tuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.dial_api.caching import get_prompt_tokens_threshold
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.adapter_deployments import (
    AdapterChatCompletionDeployment,
)
from tests.utils.openai import sanitize_test_name, sys, user

_DEPLOYMENTS = [
    D.GEMINI_2_5_PRO,
    D.GEMINI_2_5_FLASH,
    D.GEMINI_3_PRO_PREVIEW,
    D.GEMINI_3_FLASH_PREVIEW,
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
            await consumer.set_usage(TokenUsage(**usage))

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
class _TestCase:
    __test__ = False

    deployment: D
    stream: bool
    is_big_content: bool
    is_big_usage: bool | None
    caching_enabled: bool

    expected_caching_headers: bool

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

        prompt_tokens = (
            self.token_threshold
            if self.is_big_usage
            else self.token_threshold - 1
        )

        return {"prompt_tokens": prompt_tokens, "completion_tokens": 1}

    @property
    def token_threshold(self) -> int:
        return get_prompt_tokens_threshold(self.deployment) or 10

    def get_name(self):
        xs = []
        xs.append(self.deployment.value)
        xs.append("stream" if self.stream else "block")
        xs.append("caching" if self.caching_enabled else "no-caching")

        xs.append("big-content" if self.is_big_content else "small-content")

        if self.is_big_usage is not None:
            xs.append("big-usage" if self.is_big_usage else "small-usage")
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
            _TestCase(deployment, stream, True, None, True, True),
            _TestCase(deployment, stream, True, None, False, False),
            _TestCase(deployment, stream, False, None, True, False),
            _TestCase(deployment, stream, False, None, False, False),
            _TestCase(deployment, stream, True, False, True, stream),
            _TestCase(deployment, stream, True, False, False, False),
            _TestCase(deployment, stream, False, True, True, not stream),
            _TestCase(deployment, stream, False, True, False, False),
        ]
    ],
    ids=lambda x: x.get_name(),
)
async def test_dial_cache(test_http_client: httpx.AsyncClient, ts: _TestCase):
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

    has_threshold = get_prompt_tokens_threshold(ts.deployment) is not None

    if ts.expected_caching_headers and has_threshold:
        assert cache_path == "prefix.body.messages[1]"
        assert expire_at is not None
    else:
        assert cache_path is None
        assert expire_at is None
