from typing import List
from unittest.mock import Mock, call

import pytest
from vertexai.preview.language_models import ChatMessage

from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt, ChatAuthor
from aidial_adapter_vertexai.chat.bison.truncate_prompt import (
    get_discarded_messages_count,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.errors import ValidationError


@pytest.mark.asyncio
async def test_history_truncation_no_discarded_messages():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [1]

    messages: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="Hello"),
    ]

    prompt = BisonPrompt(context=None, messages=messages)

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, prompt, 1
    )

    assert discarded_messages_count == 0
    assert chat_adapter.count_prompt_tokens.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_prompt_is_too_big():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [2]

    messages: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="Hello")
    ]

    prompt = BisonPrompt(context=None, messages=messages)

    with pytest.raises(ValidationError) as exc_info:
        await get_discarded_messages_count(chat_adapter, prompt, 1)

    assert (
        str(exc_info.value)
        == "Prompt token size (2) exceeds prompt token limit (1)."
    )


@pytest.mark.asyncio
async def test_history_truncation_estimated_precisely():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 2]

    context = "message1"
    messages: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
        ChatMessage(author=ChatAuthor.USER, content="message6"),
    ]

    prompt = BisonPrompt(context=context, messages=messages)

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, prompt, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(prompt),
        call(BisonPrompt(context=context, messages=[messages[4]])),
    ]


@pytest.mark.asyncio
async def test_history_truncation_correct_but_not_exact():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 1, 2]

    context = "message1"
    messages: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
        ChatMessage(author=ChatAuthor.USER, content="message6"),
    ]

    prompt = BisonPrompt(context=context, messages=messages)

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, prompt, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(prompt),
        call(BisonPrompt(context=context, messages=[messages[4]])),
        call(BisonPrompt(context=None, messages=messages[2:4])),
    ]


@pytest.mark.asyncio
async def test_history_truncation_underestimated():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 1, 1]

    context = "message1"
    messages: List[ChatMessage] = [
        ChatMessage(author=ChatAuthor.USER, content="message2"),
        ChatMessage(author=ChatAuthor.BOT, content="message3"),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
        ChatMessage(author=ChatAuthor.USER, content="message6"),
    ]

    prompt = BisonPrompt(context=context, messages=messages)

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, prompt, 2
    )
    assert discarded_messages_count == 2
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(prompt),
        call(BisonPrompt(context=context, messages=[messages[4]])),
        call(BisonPrompt(context=None, messages=messages[2:4])),
    ]


@pytest.mark.asyncio
async def test_history_truncation_overestimated():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 4, 2]

    context = "message1"
    messages: List[ChatMessage] = [
        ChatMessage(
            author=ChatAuthor.USER,
            content="loooooooooooooooooooooooonger_message2",
        ),
        ChatMessage(
            author=ChatAuthor.BOT,
            content="loooooooooooooooooooooooonger_message3",
        ),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
        ChatMessage(author=ChatAuthor.USER, content="message6"),
    ]

    prompt = BisonPrompt(context=context, messages=messages)

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, prompt, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(prompt),
        call(BisonPrompt(context=context, messages=messages[2:5])),
        call(BisonPrompt(context=None, messages=messages[2:4])),
    ]


@pytest.mark.asyncio
async def test_history_truncation_overestimated_and_last_message_is_too_big():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 4, 1]

    context = "message1"
    messages: List[ChatMessage] = [
        ChatMessage(
            author=ChatAuthor.USER,
            content="loooooooooooooooooooooooonger_message2",
        ),
        ChatMessage(
            author=ChatAuthor.BOT,
            content="loooooooooooooooooooooooonger_message3",
        ),
        ChatMessage(author=ChatAuthor.USER, content="message4"),
        ChatMessage(author=ChatAuthor.BOT, content="message5"),
        ChatMessage(author=ChatAuthor.USER, content="message6"),
    ]

    prompt = BisonPrompt(context=context, messages=messages)

    with pytest.raises(ValidationError) as exc_info:
        await get_discarded_messages_count(chat_adapter, prompt, 2)

    assert (
        str(exc_info.value)
        == "The token size of system message and the last user message (3) exceeds prompt token limit (2)."
    )
