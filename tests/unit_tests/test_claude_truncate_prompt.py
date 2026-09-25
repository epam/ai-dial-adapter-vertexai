from typing import Any, cast

from aidial_sdk.chat_completion import Message, Role
from aidial_sdk.chat_completion.request import ChatCompletionRequest

from aidial_adapter_vertexai.chat.claude.adapter import (
    ClaudeChatCompletionAdapter,
    ClaudePrompt,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters


class _StubClaudeAdapter:
    def __init__(self, discarded: list[int]) -> None:
        self.discarded = discarded
        self.seen_max_prompt_tokens: int | None = None

    async def compute_discarded_messages(self, request) -> list[int]:
        self.seen_max_prompt_tokens = request.max_prompt_tokens
        return self.discarded


async def _parse(request: ChatCompletionRequest, claude_adapter: Any):
    adapter = ClaudeChatCompletionAdapter(
        claude_adapter=cast(Any, claude_adapter)
    )
    prompt = await adapter.parse_prompt(
        ModelParameters(),
        ToolsConfig.noop(),
        StaticToolsConfig.noop(),
        request,
    )
    assert isinstance(prompt, ClaudePrompt)
    return adapter, prompt


async def test_truncate_prompt_omits_discarded_messages():
    request = ChatCompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="system"),
            Message(role=Role.USER, content="one"),
            Message(role=Role.ASSISTANT, content="two"),
            Message(role=Role.USER, content="three"),
        ]
    )
    claude_adapter = _StubClaudeAdapter(discarded=[1, 2])
    adapter, prompt = await _parse(request, claude_adapter)

    truncated = await adapter.truncate_prompt(prompt, 100)

    # the limit reaches the underlying adapter, which only reads it off the request
    assert claude_adapter.seen_max_prompt_tokens == 100

    assert truncated.discarded_messages == [1, 2]
    assert [m.content for m in truncated.prompt.dial_request.messages] == [
        "system",
        "three",
    ]
    assert len(truncated.prompt.request.messages) == 2
