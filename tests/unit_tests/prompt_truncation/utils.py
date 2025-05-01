from typing import Awaitable, Callable, TypeVar

from aidial_adapter_vertexai.chat.truncate_prompt import (
    DiscardedMessages,
    TruncatablePrompt,
)

_P = TypeVar("_P", bound=TruncatablePrompt)


async def get_discarded_messages(
    tokenize: Callable[[_P], Awaitable[int]],
    prompt: _P,
    max_prompt_tokens: int,
) -> DiscardedMessages:
    return (
        await prompt.truncate(tokenize=tokenize, user_limit=max_prompt_tokens)
    ).discarded_messages
