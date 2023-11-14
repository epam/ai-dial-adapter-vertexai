from typing import List
from unittest.mock import Mock, call

import pytest

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.llm.history_trimming import (
    get_discarded_messages_count,
)
from aidial_adapter_vertexai.llm.vertex_ai_chat import (
    VertexAIAuthor,
    VertexAIMessage,
)


@pytest.mark.asyncio
async def test_history_truncation_no_discarded_messages():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [1]

    messages: List[VertexAIMessage] = [
        VertexAIMessage(author=VertexAIAuthor.USER, content="Hello"),
    ]

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, None, messages, 1
    )

    assert discarded_messages_count == 0
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(None, messages)
    ]


@pytest.mark.asyncio
async def test_history_truncation_prompt_is_too_big():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [2]

    messages: List[VertexAIMessage] = [
        VertexAIMessage(author=VertexAIAuthor.USER, content="Hello")
    ]

    with pytest.raises(ValidationError) as exc_info:
        await get_discarded_messages_count(chat_adapter, None, messages, 1)

    assert (
        str(exc_info.value)
        == "Prompt token size (2) exceeds prompt token limit (1)."
    )


@pytest.mark.asyncio
async def test_history_truncation_estimated_precisely():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 2]

    context = "message1"
    messages: List[VertexAIMessage] = [
        VertexAIMessage(author=VertexAIAuthor.USER, content="message2"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message3"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message4"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message5"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message6"),
    ]

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, context, messages, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(context, messages),
        call(context, [messages[4]]),
    ]


@pytest.mark.asyncio
async def test_history_truncation_correct_but_not_exact():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 1, 2]

    context = "message1"
    messages: List[VertexAIMessage] = [
        VertexAIMessage(author=VertexAIAuthor.USER, content="message2"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message3"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message4"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message5"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message6"),
    ]

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, context, messages, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(context, messages),
        call(context, [messages[4]]),
        call(None, messages[2:4]),
    ]


@pytest.mark.asyncio
async def test_history_truncation_underestimated():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 1, 1]

    context = "message1"
    messages: List[VertexAIMessage] = [
        VertexAIMessage(author=VertexAIAuthor.USER, content="message2"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message3"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message4"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message5"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message6"),
    ]

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, context, messages, 2
    )
    assert discarded_messages_count == 2
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(context, messages),
        call(context, [messages[4]]),
        call(None, messages[2:4]),
    ]


@pytest.mark.asyncio
async def test_history_truncation_overestimated():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 4, 2]

    context = "message1"
    messages: List[VertexAIMessage] = [
        VertexAIMessage(
            author=VertexAIAuthor.USER,
            content="loooooooooooooooooooooooonger_message2",
        ),
        VertexAIMessage(
            author=VertexAIAuthor.BOT,
            content="loooooooooooooooooooooooonger_message3",
        ),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message4"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message5"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message6"),
    ]

    discarded_messages_count = await get_discarded_messages_count(
        chat_adapter, context, messages, 2
    )
    assert discarded_messages_count == 4
    assert chat_adapter.count_prompt_tokens.call_args_list == [
        call(context, messages),
        call(context, messages[2:5]),
        call(None, messages[2:4]),
    ]


@pytest.mark.asyncio
async def test_history_truncation_overestimated_and_last_message_is_too_big():
    chat_adapter = Mock(spec=ChatCompletionAdapter)
    chat_adapter.count_prompt_tokens.side_effect = [6, 4, 1]

    context = "message1"
    messages: List[VertexAIMessage] = [
        VertexAIMessage(
            author=VertexAIAuthor.USER,
            content="loooooooooooooooooooooooonger_message2",
        ),
        VertexAIMessage(
            author=VertexAIAuthor.BOT,
            content="loooooooooooooooooooooooonger_message3",
        ),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message4"),
        VertexAIMessage(author=VertexAIAuthor.BOT, content="message5"),
        VertexAIMessage(author=VertexAIAuthor.USER, content="message6"),
    ]

    with pytest.raises(ValidationError) as exc_info:
        await get_discarded_messages_count(chat_adapter, context, messages, 2)

    assert (
        str(exc_info.value)
        == "The token size of system message and the last user message (3) exceeds prompt token limit (2)."
    )
