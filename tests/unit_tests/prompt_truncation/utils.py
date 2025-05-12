from typing import Awaitable, Callable, Type, TypeVar

from google.genai.types import Content as GenAIContent

from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiBasePrompt
from aidial_adapter_vertexai.chat.truncate_prompt import (
    DiscardedMessages,
    TruncatablePrompt,
)
from aidial_adapter_vertexai.utils.list import MessageMergeStrategy

_P = TypeVar("_P", bound=TruncatablePrompt)


async def get_discarded_messages(
    tokenize: Callable[[_P], Awaitable[int]],
    prompt: _P,
    max_prompt_tokens: int,
) -> DiscardedMessages:
    return (
        await prompt.get_truncated_prompt(
            tokenize=tokenize, user_limit=max_prompt_tokens
        )
    ).discarded_messages


async def get_discarded_messages_with_message_merge(
    tokenize: Callable[[GeminiBasePrompt], Awaitable[int]],
    prompt: GeminiBasePrompt,
    merger: Type[MessageMergeStrategy[GenAIContent]],
    max_prompt_tokens: int,
) -> DiscardedMessages:
    prompt.conversation = prompt.conversation.merge_messages_with_same_role(
        merger
    )
    prompt = await prompt.truncate(
        tokenize=tokenize, user_limit=max_prompt_tokens
    )
    return list(prompt.conversation.messages.get_removed_indices())
