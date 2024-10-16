from typing import List, TypeVar, cast
from unittest.mock import Mock, call

import pytest
from aidial_sdk.exceptions import HTTPException as DialException
from vertexai.preview.language_models import ChatMessage

from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt, ChatAuthor
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.truncate_prompt import (
    DiscardedMessages,
    Truncatable,
)

_P = TypeVar("_P", bound=Truncatable)


async def get_discarded_messages(
    model: ChatCompletionAdapter[_P], prompt: _P, max_prompt_tokens: int
) -> DiscardedMessages:
    discarded_indices, _ = await prompt.truncate_prompt(
        tokenizer=model.count_prompt_tokens,
        user_limit=max_prompt_tokens,
    )

    return discarded_indices


@pytest.fixture()
def mock_model():
    async def count_tokens_by_words(prompt: BisonPrompt) -> int:
        text = " ".join(
            [
                prompt.context or "",
                *[msg.content for msg in prompt.history],
                prompt.prompt,
            ]
        )
        return len(text.split())

    model = Mock(spec=ChatCompletionAdapter)
    model.count_prompt_tokens.side_effect = count_tokens_by_words
    yield cast(ChatCompletionAdapter[BisonPrompt], model)


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_1(mock_model):

    prompt = BisonPrompt(prompt="hello")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 1)

    assert discarded_messages == []
    assert mock_model.count_prompt_tokens.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_2(mock_model):

    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=None, history=history, prompt="message6")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 5)

    assert discarded_messages == []
    assert mock_model.count_prompt_tokens.call_args_list == [
        call(prompt.omit({0, 1, 2, 3})),
        call(prompt.omit({0, 1})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_3(mock_model):

    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=None, history=history, prompt="message6")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 1000)

    assert discarded_messages == []
    assert mock_model.count_prompt_tokens.call_args_list == [
        call(prompt.omit({0, 1, 2, 3})),
        call(prompt.omit({0, 1})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_all_turns(mock_model):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=context, history=history, prompt="message6")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 2)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_model.count_prompt_tokens.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_mid_turn(mock_model):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=context, history=history, prompt="message6")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 3)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_model.count_prompt_tokens.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_last_turn(mock_model):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=context, history=history, prompt="message6")

    discarded_messages = await get_discarded_messages(mock_model, prompt, 4)
    assert discarded_messages == [1, 2]
    assert mock_model.count_prompt_tokens.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_last_and_system_messages_are_too_big(
    mock_model,
):
    context = "message1 message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(context=context, history=history, prompt="message6")

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_model, prompt, 2)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 2. However, the system messages and the last user message resulted in 3 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_history_truncation_last_message_is_too_big(mock_model):
    prompt = BisonPrompt(prompt="hello hello")

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_model, prompt, 1)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in 2 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )
