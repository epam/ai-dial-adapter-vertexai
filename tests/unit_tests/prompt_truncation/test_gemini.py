from typing import List
from unittest.mock import AsyncMock, call

import pytest
from aidial_sdk.exceptions import HTTPException as DialException
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    Gemini_1_5_Prompt,
)
from tests.unit_tests.prompt_truncation.utils import get_discarded_messages


async def tokenize_by_words(prompt: GeminiPrompt) -> int:
    text = " ".join(
        [
            *[part.text for part in prompt.system_instruction or []],
            *[msg.text for msg in prompt.contents],
        ]
    )
    return len(text.split())


def text_parts(s: str) -> List[Part]:
    return [Part.from_text(s)]


def sys(s: str) -> List[Part]:
    return text_parts(s)


def user(s: str) -> Content:
    return Content(role=ChatSession._USER_ROLE, parts=text_parts(s))


def bot(s: str) -> Content:
    return Content(role=ChatSession._MODEL_ROLE, parts=text_parts(s))


@pytest.fixture
def mock_tokenize():
    mock = AsyncMock()
    mock.side_effect = tokenize_by_words
    return mock


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_1(mock_tokenize):

    prompt = Gemini_1_5_Prompt(contents=[user("hello")])

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 1)

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_2(mock_tokenize):

    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=None,
        contents=contents,
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 5)

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_cut_nothing_3(mock_tokenize):

    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=None,
        contents=contents,
    )

    discarded_messages = await get_discarded_messages(
        mock_tokenize, prompt, 1000
    )

    assert discarded_messages == []
    assert mock_tokenize.call_args_list == [call(prompt)]


@pytest.mark.asyncio
async def test_history_truncation_cut_all_turns(mock_tokenize):
    system = sys("message1")
    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=system,
        contents=contents,
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 2)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_tokenize.call_args_list == [
        call(prompt),
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_mid_turn(mock_tokenize):
    system = sys("message1")
    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=system,
        contents=contents,
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 3)
    assert discarded_messages == [1, 2, 3, 4]
    assert mock_tokenize.call_args_list == [
        call(prompt),
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_cut_last_turn(mock_tokenize):
    system = sys("message1")
    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=system,
        contents=contents,
    )

    discarded_messages = await get_discarded_messages(mock_tokenize, prompt, 4)
    assert discarded_messages == [1, 2]
    assert mock_tokenize.call_args_list == [
        call(prompt),
        call(prompt.omit({1, 2, 3, 4})),
        call(prompt.omit({1, 2})),
    ]


@pytest.mark.asyncio
async def test_history_truncation_last_and_system_messages_are_too_big(
    mock_tokenize,
):
    system = sys("message1 message1")
    contents: List[Content] = [
        user("message2"),
        bot("message3"),
        user("message4"),
        bot("message5"),
        user("message6"),
    ]

    prompt = Gemini_1_5_Prompt(
        system_instruction=system,
        contents=contents,
    )

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_tokenize, prompt, 2)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 2. However, the system messages and the last user message resulted in 3 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_history_truncation_last_message_is_too_big(mock_tokenize):
    prompt = Gemini_1_5_Prompt(contents=[user("hello hello")])

    with pytest.raises(DialException) as exc_info:
        await get_discarded_messages(mock_tokenize, prompt, 1)

    assert (
        exc_info.value.message
        == "The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in 2 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )
