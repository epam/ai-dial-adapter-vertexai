import base64

import pytest
from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_transcribe import (
    GeminiTranscribePrompt,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.utils.resource import Resource

_TEXT_INPUT_ERROR = "Transcription models don't support text input."


def _audio_message(payload: bytes, text: str | None = None) -> Message:
    text_parts = [] if text is None else [{"type": "text", "text": text}]
    return Message.model_validate(
        {
            "role": "user",
            "content": [
                *text_parts,
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(payload).decode(),
                        "format": "mp3",
                    },
                },
            ],
        }
    )


async def _parse(messages: list[Message]) -> GeminiTranscribePrompt | UserError:
    return await GeminiTranscribePrompt.parse(
        file_storage=None,
        tools=ToolsConfig.noop(),
        static_tools=StaticToolsConfig.noop(),
        messages=messages,
    )


async def test_transcribe_prompt_keeps_only_last_message():
    messages = [
        _audio_message(b"first-audio"),
        Message.model_validate(
            {"role": "assistant", "content": "first transcript"}
        ),
        _audio_message(b"second-audio"),
    ]

    prompt = await _parse(messages)

    assert isinstance(prompt, GeminiTranscribePrompt)
    contents = list(prompt.messages.raw_list)
    assert len(contents) == 1
    assert contents[0].role == "user"
    audio_parts = [p for p in contents[0].parts or [] if p.inline_data]
    assert len(audio_parts) == 1
    assert audio_parts[0].inline_data
    assert audio_parts[0].inline_data.data == b"second-audio"
    assert audio_parts[0].inline_data.mime_type == "audio/mp3"


async def test_transcribe_prompt_rejects_system_message():
    messages = [
        Message.model_validate({"role": "system", "content": "Be concise"}),
        _audio_message(b"audio"),
    ]

    with pytest.raises(ValidationError, match="System messages"):
        await _parse(messages)


async def test_transcribe_prompt_requires_audio():
    messages = [Message.model_validate({"role": "user", "content": ""})]

    prompt = await _parse(messages)

    assert isinstance(prompt, UserError)
    assert prompt.error_message == "Attach an audio file to transcribe"


async def test_transcribe_prompt_rejects_text_only_message():
    messages = [Message.model_validate({"role": "user", "content": "Hello"})]

    prompt = await _parse(messages)

    assert isinstance(prompt, UserError)
    assert prompt.error_message == _TEXT_INPUT_ERROR
    assert prompt.to_dial_exception().display_message == _TEXT_INPUT_ERROR


async def test_transcribe_prompt_rejects_text_with_audio():
    messages = [_audio_message(b"audio", text="Transcribe in French")]

    prompt = await _parse(messages)

    assert isinstance(prompt, UserError)
    assert prompt.error_message == _TEXT_INPUT_ERROR


async def test_transcribe_prompt_allows_blank_text_with_audio():
    messages = [_audio_message(b"audio", text="  ")]

    prompt = await _parse(messages)

    assert isinstance(prompt, GeminiTranscribePrompt)


async def test_transcribe_prompt_ignores_text_in_previous_messages():
    messages = [
        Message.model_validate({"role": "user", "content": "Hello"}),
        Message.model_validate({"role": "assistant", "content": "Hi"}),
        _audio_message(b"audio"),
    ]

    prompt = await _parse(messages)

    assert isinstance(prompt, GeminiTranscribePrompt)


async def test_transcribe_prompt_rejects_non_audio_attachment():
    image_url = Resource(type="image/png", data=b"image").to_data_url()
    messages = [
        Message.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}}
                ],
            }
        )
    ]

    prompt = await _parse(messages)

    assert isinstance(prompt, UserError)
    assert "failed to process" in prompt.error_message
    assert prompt.usage_message and "mp3" in prompt.usage_message
