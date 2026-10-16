from typing import Any, cast

from google.genai.types import GenerateContentResponseUsageMetadata

from aidial_adapter_vertexai.chat.gemini.output import set_usage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage


class UsageConsumer:
    usage: TokenUsage | None = None

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage


async def test_tool_use_prompt_tokens_are_counted_as_prompt_tokens():
    # Real usage reported by gemini-2.5-pro with the code execution tool.
    # The cached tokens cover both the prompt and the tool use prompt.
    usage = GenerateContentResponseUsageMetadata(
        prompt_token_count=23960,
        cached_content_token_count=45452,
        tool_use_prompt_token_count=24824,
        thoughts_token_count=274,
        candidates_token_count=1745,
        total_token_count=50803,
    )

    consumer = UsageConsumer()
    await set_usage(usage, cast(Any, consumer), cast(Any, None), False)

    assert (actual := consumer.usage) is not None
    assert actual == TokenUsage(
        prompt_tokens=23960 + 24824,
        prompt_cached_tokens=45452,
        completion_tokens=274 + 1745,
        completion_reasoning_tokens=274,
    )
    assert actual.total_tokens == usage.total_token_count
