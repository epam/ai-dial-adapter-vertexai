from typing import List, Tuple

from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.errors import ValidationError


def _estimate_discarded_messages(
    prompt: BisonPrompt, prompt_tokens: int, max_prompt_tokens: int
) -> int:
    context, messages = prompt.context, prompt.messages

    text_size = len(context or "") + sum(len(m.content) for m in messages)
    estimated_token_size: float = text_size / prompt_tokens

    discarded_messages = 0
    for index in range(0, len(messages) - 1, 2):
        text_size -= len(messages[index].content + messages[index + 1].content)
        discarded_messages += 2

        if text_size / estimated_token_size <= max_prompt_tokens:
            break

    return discarded_messages


async def get_discarded_messages_count(
    model: ChatCompletionAdapter[BisonPrompt],
    prompt: BisonPrompt,
    max_prompt_tokens: int,
) -> int:
    context, messages = prompt.context, prompt.messages

    prompt_tokens = await model.count_prompt_tokens(prompt)
    if prompt_tokens <= max_prompt_tokens or prompt_tokens == 0:
        return 0

    if len(messages) == 1:
        raise ValidationError(
            f"Prompt token size ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})."
        )

    discarded_messages_count = _estimate_discarded_messages(
        prompt, prompt_tokens, max_prompt_tokens
    )

    prompt_tokens = await model.count_prompt_tokens(
        BisonPrompt(
            context=context, messages=messages[discarded_messages_count:]
        )
    )

    if prompt_tokens == max_prompt_tokens:
        return discarded_messages_count
    elif prompt_tokens > max_prompt_tokens:
        for index in range(discarded_messages_count, len(messages) - 1, 2):
            prompt_tokens -= await model.count_prompt_tokens(
                BisonPrompt(context=None, messages=messages[index : index + 2])
            )
            discarded_messages_count += 2

            if prompt_tokens <= max_prompt_tokens:
                return discarded_messages_count

        raise ValidationError(
            f"The token size of system message and the last user message ({prompt_tokens}) exceeds"
            f" prompt token limit ({max_prompt_tokens})."
        )
    else:  # prompt_tokens < max_prompt_tokens
        for index in range(discarded_messages_count - 2, 0, -2):
            prompt_tokens += await model.count_prompt_tokens(
                BisonPrompt(context=None, messages=messages[index : index + 2])
            )
            if prompt_tokens > max_prompt_tokens:
                break

            discarded_messages_count -= 2

        return discarded_messages_count


async def get_discarded_messages(
    model: ChatCompletionAdapter[BisonPrompt],
    prompt: BisonPrompt,
    max_prompt_tokens: int,
) -> Tuple[BisonPrompt, List[int]]:
    count = await get_discarded_messages_count(model, prompt, max_prompt_tokens)

    truncated_prompt = BisonPrompt(
        context=prompt.context,
        messages=prompt.messages[count:],
    )

    discarded_indices = list(range(count))
    if prompt.context is not None:
        discarded_indices = list(map(lambda x: x + 1, discarded_indices))

    return truncated_prompt, discarded_indices
