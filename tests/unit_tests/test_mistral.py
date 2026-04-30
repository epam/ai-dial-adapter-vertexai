import json
from typing import Any, cast

import pytest
from aidial_sdk.chat_completion import FinishReason, Message
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from mistralai.gcp.client.models import FunctionCall, ToolCall

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.mistral.adapter import (
    CompletionChunk,
    Consumer,
    _append_tool_calls_state,
    _consume_stream_chunk,
    _consume_tool_calls,
    _to_finish_reason,
    _to_prompt_mode,
)
from aidial_adapter_vertexai.chat.mistral.prompt import (
    MistralPromptParser,
    _file_content_part_to_image_url_chunk,
    _normalize_tool_description,
    _resolve_local_json_refs,
    _to_mistral_tool_call_id,
)
from aidial_adapter_vertexai.chat.mistral.state import _ToolCallState
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.utils.resource import Resource


class _ConsumerStub:
    def __init__(self):
        self.finish_reason = None
        self.content: list[str] = []

    async def set_finish_reason(self, finish_reason: FinishReason) -> None:
        self.finish_reason = finish_reason


class _ToolConsumerStub:
    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []
        self.has_function_call = False

    async def create_tool_call(
        self, id: str, name: str, arguments: str | None
    ) -> None:
        self.calls.append((id, name, arguments))


class _ChunkDeltaStub:
    def __init__(self, content: str | None, tool_calls: Any):
        self.content = content
        self.tool_calls = tool_calls


class _ChunkChoiceStub:
    def __init__(self, delta: _ChunkDeltaStub, finish_reason: str):
        self.delta = delta
        self.finish_reason = finish_reason


class _ChunkStub:
    def __init__(self, choices: list[_ChunkChoiceStub]):
        self.choices = choices


def _request_with_reasoning_effort(
    reasoning_effort: str | None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": reasoning_effort,
        }
    )


def test_resolve_local_json_refs_inlines_defs_and_keeps_siblings():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "$ref": "#/$defs/User",
                "description": "Resolved user",
            }
        },
        "$defs": {
            "User": {"type": "object", "properties": {"id": {"type": "string"}}}
        },
    }

    resolved = _resolve_local_json_refs(schema)

    assert "$defs" not in resolved
    assert resolved["properties"]["user"]["type"] == "object"
    assert (
        resolved["properties"]["user"]["properties"]["id"]["type"] == "string"
    )
    assert resolved["properties"]["user"]["description"] == "Resolved user"


def test_to_mistral_tool_call_id_maps_invalid_values_stably():
    id_map: dict[str, str] = {}

    first = _to_mistral_tool_call_id("call_1", id_map=id_map)
    second = _to_mistral_tool_call_id("call_1", id_map=id_map)
    third = _to_mistral_tool_call_id("another", id_map=id_map)

    assert first == "tc0000000"
    assert second == first
    assert third == "tc0000001"


def test_to_mistral_tool_call_id_keeps_valid_mistral_id():
    id_map: dict[str, str] = {}

    tool_call_id = _to_mistral_tool_call_id("AbC123xY9", id_map=id_map)

    assert tool_call_id == "AbC123xY9"


def test_normalize_tool_description_uses_default_for_empty_values():
    assert _normalize_tool_description(None) == "Tool function"
    assert _normalize_tool_description("  ") == "Tool function"
    assert _normalize_tool_description("Meaningful") == "Meaningful"


def test_file_content_part_to_image_url_chunk_supports_legacy_pdf_base64():
    payload = b"%PDF-fake"
    base64_pdf = Resource(type="application/pdf", data=payload).data_base64
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {"file_data": base64_pdf, "filename": "doc.pdf"},
                }
            ],
        }
    )
    assert message.content is not None
    file_part = cast(Any, message.content[0])

    chunk = _file_content_part_to_image_url_chunk(
        file_part.file, content_name="User message content"
    )

    image_url = chunk.image_url
    url = image_url if isinstance(image_url, str) else image_url.url
    assert isinstance(url, str)
    assert url.startswith("data:application/pdf;base64,")


def test_file_content_part_to_image_url_chunk_rejects_unsupported_types():
    text_data_url = Resource(type="text/plain", data=b"abc").to_data_url()
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": text_data_url,
                        "filename": "data.txt",
                    },
                }
            ],
        }
    )
    assert message.content is not None
    file_part = cast(Any, message.content[0])

    with pytest.raises(ValidationError, match="unsupported file content part"):
        _file_content_part_to_image_url_chunk(
            file_part.file, content_name="User message content"
        )


def test_tool_call_state_to_tool_call_parses_json_arguments():
    state = _ToolCallState(index=3, id="id1234567", name="search")
    state.arguments = json.dumps({"q": "python"})

    tool_call = state.to_tool_call()

    assert tool_call.index == 3
    assert tool_call.id == "id1234567"
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == {"q": "python"}


def test_tool_call_state_to_tool_call_requires_name():
    state = _ToolCallState(index=0, id="id1234567", name=None, arguments="{}")

    with pytest.raises(ValidationError, match="function name is missing"):
        state.to_tool_call()


def test_append_tool_calls_state_accumulates_streamed_arguments():
    state: dict[int, _ToolCallState] = {}
    chunks = [
        ToolCall(
            id="id1234567",
            index=0,
            function=FunctionCall(name="search", arguments='{"q":"py'),
        ),
        ToolCall(
            id="id1234567",
            index=0,
            function=FunctionCall(name="search", arguments='thon"}'),
        ),
    ]

    _append_tool_calls_state(state, chunks)

    assert state[0].name == "search"
    assert state[0].arguments == '{"q":"python"}'


async def test_consume_stream_chunk_rewrites_tool_calls_finish_without_tools():
    chunk = _ChunkStub(
        choices=[
            _ChunkChoiceStub(
                delta=_ChunkDeltaStub(content=None, tool_calls=None),
                finish_reason="tool_calls",
            )
        ]
    )
    consumer = _ConsumerStub()

    finish_reason = await _consume_stream_chunk(
        cast(CompletionChunk, chunk),
        cast(Consumer, consumer),
        tool_calls_state={},
        allow_tool_calls=False,
    )

    assert finish_reason == FinishReason.STOP
    assert consumer.finish_reason == FinishReason.STOP


def test_to_finish_reason_maps_mistral_specific_values():
    assert _to_finish_reason("length") == FinishReason.LENGTH
    assert _to_finish_reason("model_length") == FinishReason.LENGTH
    assert _to_finish_reason("tool_calls") == FinishReason.TOOL_CALLS
    assert _to_finish_reason("error") == FinishReason.CONTENT_FILTER
    assert _to_finish_reason("anything_else") == FinishReason.STOP


def test_to_prompt_mode_uses_reasoning_for_non_none_effort():
    no_reasoning = ModelParameters.create(
        _request_with_reasoning_effort("none")
    )
    medium_reasoning = ModelParameters.create(
        _request_with_reasoning_effort("medium")
    )

    assert _to_prompt_mode(no_reasoning) is None
    assert _to_prompt_mode(medium_reasoning) == "reasoning"


@pytest.mark.asyncio
async def test_parse_legacy_function_messages_reuse_same_tool_call_id():
    messages = [
        Message.model_validate({"role": "user", "content": "hi"}),
        Message.model_validate(
            {
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": "search",
                    "arguments": '{"q":"python"}',
                },
            }
        ),
        Message.model_validate(
            {
                "role": "function",
                "name": "search",
                "content": '{"result":"ok"}',
            }
        ),
    ]
    params = ModelParameters.create(_request_with_reasoning_effort(None))

    prompt = await MistralPromptParser.parse(
        params=params,
        tools=ToolsConfig.noop(),
        file_storage=None,
        messages=messages,
    )

    assistant_message = prompt.messages[1]
    function_result_message = prompt.messages[2]

    assistant_tool_call = assistant_message.tool_calls[0]
    assert assistant_tool_call.id == "fc0000000"
    assert function_result_message.tool_call_id == assistant_tool_call.id


@pytest.mark.asyncio
async def test_parse_tool_message_keeps_original_content():
    messages = [
        Message.model_validate({"role": "user", "content": "hi"}),
        Message.model_validate(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123xyz",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"python"}',
                        },
                    }
                ],
            }
        ),
        Message.model_validate(
            {
                "role": "tool",
                "tool_call_id": "abc123xyz",
                "content": '{"result":"ok"}',
            }
        ),
    ]
    params = ModelParameters.create(_request_with_reasoning_effort(None))

    prompt = await MistralPromptParser.parse(
        params=params,
        tools=ToolsConfig.noop(),
        file_storage=None,
        messages=messages,
    )

    tool_message = prompt.messages[2]
    assert tool_message.content == '{"result":"ok"}'
    assert tool_message.name is None


@pytest.mark.asyncio
async def test_consume_tool_calls_preserves_upstream_tool_call_id():
    consumer = _ToolConsumerStub()
    tool_calls = [
        ToolCall(
            id="AbC123xY9",
            index=0,
            function=FunctionCall(
                name="get_temperature", arguments='{"location":"London"}'
            ),
        )
    ]

    await _consume_tool_calls(
        tool_calls,
        cast(Consumer, consumer),
        use_tool_api=True,
        allow_tool_calls=True,
    )

    assert consumer.calls == [
        ("AbC123xY9", "get_temperature", '{"location":"London"}')
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_calls", "expected_calls"),
    [
        (
            [
                ToolCall(
                    id="null",
                    index=0,
                    function=FunctionCall(
                        name="get_temperature",
                        arguments='{"location":"London"}',
                    ),
                )
            ],
            [("get_temperature_1", "get_temperature", '{"location":"London"}')],
        ),
        (
            [
                ToolCall(
                    id="",
                    index=0,
                    function=FunctionCall(name="tool_a", arguments='{"x":1}'),
                ),
                ToolCall(
                    id="null",
                    index=1,
                    function=FunctionCall(name="tool_b", arguments='{"y":2}'),
                ),
            ],
            [
                ("tool_a_1", "tool_a", '{"x":1}'),
                ("tool_b_2", "tool_b", '{"y":2}'),
            ],
        ),
        (
            [
                ToolCall(
                    id="AbC123xY9",
                    index=0,
                    function=FunctionCall(name="tool_a", arguments='{"a":1}'),
                ),
                ToolCall(
                    id="null",
                    index=1,
                    function=FunctionCall(name="tool_b", arguments='{"b":2}'),
                ),
                ToolCall(
                    id="",
                    index=2,
                    function=FunctionCall(name="tool_c", arguments='{"c":3}'),
                ),
            ],
            [
                ("AbC123xY9", "tool_a", '{"a":1}'),
                ("tool_b_2", "tool_b", '{"b":2}'),
                ("tool_c_3", "tool_c", '{"c":3}'),
            ],
        ),
    ],
)
async def test_consume_tool_calls_preserve_order_and_assign_fallback_ids(
    tool_calls: list[ToolCall],
    expected_calls: list[tuple[str, str, str | None]],
):
    consumer = _ToolConsumerStub()

    await _consume_tool_calls(
        tool_calls,
        cast(Consumer, consumer),
        use_tool_api=True,
        allow_tool_calls=True,
    )

    assert consumer.calls == expected_calls
