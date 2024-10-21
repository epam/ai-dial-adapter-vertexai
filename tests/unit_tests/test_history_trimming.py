from typing import Awaitable, Callable, List, TypeVar
from unittest.mock import AsyncMock, call

import pytest
from aidial_sdk.exceptions import HTTPException as DialException
from vertexai.preview.language_models import ChatMessage

from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt, ChatAuthor
from aidial_adapter_vertexai.chat.truncate_prompt import (
    DiscardedMessages,
    TruncatablePrompt,
)

_P = TypeVar("_P", bound=TruncatablePrompt)


async def get_discarded_messages(
    tokenizer: Callable[[_P], Awaitable[int]],
    prompt: _P,
    max_prompt_tokens: int,
) -> DiscardedMessages:
    return (
        await prompt.truncate(tokenizer=tokenizer, user_limit=max_prompt_tokens)
    )[0]


class MockBisonPrompt(BisonPrompt):
    async def tokenize_by_words(self) -> int:
        text = " ".join(
            [
                self.system_instruction or "",
                *[msg.content for msg in self.history],
                self.last_user_message,
            ]
        )
        return len(text.split())


@pytest.fixture
def mock_tokenize():
    mock = AsyncMock()
    mock.side_effect = MockBisonPrompt.tokenize_by_words
    return mock


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_1(mock_tokenize):

    prompt = BisonPrompt(last_user_message="hello")

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 1)

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_2(mock_tokenize):

    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=None,
        history=history,
        last_user_message="message6",
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 5)

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [
        call(prompt.omit({0, 1, 2, 3})),
        call(prompt.omit({0, 1})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_3(mock_tokenize):

    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=None,
        history=history,
        last_user_message="message6",
    )

    discarded_messages = await get_discarded_messages(
        mock_tokenize, prompt, 1000
    )

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [
        call(prompt.omit({0, 1, 2, 3})),
        call(prompt.omit({0, 1})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_all_turns(mock_tokenize):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=context,
        history=history,
        last_user_message="message6",
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 2)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_tokenize.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_mid_turn(mock_tokenize):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=context,
        history=history,
        last_user_message="message6",
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 3)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_tokenize.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_last_turn(mock_tokenize):
    context = "message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=context,
        history=history,
        last_user_message="message6",
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 4)
    assert discarded_messages == [1, 2]
    assert mock_tokenize.call_args_list == [
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
        call(prompt),
    ]


@pytest.mark.asyncio
async def test_history_truncation_last_and_system_messages_are_too_big(
    mock_tokenize,
):
    context = "message1 message1"
    history: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
    ]

    prompt = BisonPrompt(
        system_instruction=context,
        history=history,
        last_user_message="message6",
    )

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_tokenize, prompt, 2)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 2. However, the system messages and the last user message resulted in 3 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_history_truncation_last_message_is_too_big(mock_tokenize):
    prompt = BisonPrompt(last_user_message="hello hello")

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_tokenize, prompt, 1)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in 2 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )
