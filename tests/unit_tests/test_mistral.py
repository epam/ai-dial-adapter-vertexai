import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aidial_sdk.chat_completion import (
    Attachment,
    FinishReason,
    Message,
    Request,
    Response,
)
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    CustomContent,
    Role,
)
from aidial_sdk.chat_completion.request import (
    FunctionCall as DialFunctionCall,
)
from aidial_sdk.chat_completion.request import (
    ToolCall as DialToolCall,
)
from mistralai.client.models import (
    CompletionChunk,
    CompletionResponseStreamChoice,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UsageInfo,
    UserMessage,
)

from aidial_adapter_vertexai.chat.consumer import ChoiceConsumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.mistral.adapter import (
    MistralChatCompletionAdapter,
    append_tool_calls_state,
    consume_stream_chunk,
    consume_tool_calls,
    to_finish_reason,
)
from aidial_adapter_vertexai.chat.mistral.prompt import (
    MistralPrompt,
    MistralPromptParser,
    inline_local_json_refs,
    to_mistral_tool_call_id,
)
from aidial_adapter_vertexai.chat.mistral.state import ToolCallState
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.utils.adapter_deployment import AdapterDeployment
from aidial_adapter_vertexai.utils.resource import Resource


def _make_consumer_response() -> tuple[ChoiceConsumer, Response]:
    response = Response(Request.model_construct(stream=True, n=1))
    return ChoiceConsumer(response=response), response


def _extract_tool_calls(
    response: Response,
) -> list[tuple[str, str, str | None]]:
    calls: list[tuple[str, str, str | None]] = []
    while not response._queue.empty():
        chunk = cast(Any, response._queue.get_nowait()).to_dict()
        response._queue.task_done()
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            for tool_call in delta.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                calls.append(
                    (
                        tool_call.get("id"),
                        function.get("name"),
                        function.get("arguments"),
                    )
                )
    return calls


def _extract_function_calls(response: Response) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []
    while not response._queue.empty():
        chunk = cast(Any, response._queue.get_nowait()).to_dict()
        response._queue.task_done()
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            function_call = delta.get("function_call")
            if function_call:
                calls.append(
                    (
                        function_call.get("name"),
                        function_call.get("arguments"),
                    )
                )
    return calls


def _make_model_params(*, stream: bool) -> ModelParameters:
    return ModelParameters.create(
        ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="hello")],
            stream=stream,
        )
    )


class _FakeEventStream:
    def __init__(self, chunks: list[CompletionChunk]):
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return SimpleNamespace(data=chunk)


class _FakeStreamContext:
    def __init__(self, chunks: list[CompletionChunk]):
        self._chunks = chunks

    async def __aenter__(self):
        return _FakeEventStream(self._chunks)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeMistralChat:
    def __init__(
        self,
        *,
        complete_response: Any | None = None,
        stream_chunks: list[CompletionChunk] | None = None,
    ):
        self.complete_response = complete_response
        self.stream_chunks = stream_chunks or []

    async def complete_async(self, **kwargs):
        return self.complete_response

    async def stream_async(self, **kwargs):
        return _FakeStreamContext(self.stream_chunks)


def _make_adapter(
    *, chat_client: _FakeMistralChat
) -> MistralChatCompletionAdapter:
    return MistralChatCompletionAdapter(
        file_storage=None,
        deployment=AdapterDeployment(
            upstream_deployment_id="mistral-small-2503",
            reference_deployment_id=ChatCompletionDeployment.MISTRAL_SMALL,
        ),
        client=cast(Any, SimpleNamespace(chat=chat_client)),
    )


def _make_prompt(
    *, tools_enabled: bool, use_tool_api: bool = True
) -> MistralPrompt:
    return MistralPrompt(
        messages=[UserMessage(content="hello")],
        tools=[] if tools_enabled else None,
        use_tool_api=use_tool_api,
    )


def test_inline_json_references():
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

    resolved = inline_local_json_refs(schema)

    assert resolved == {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "description": "Resolved user",
            }
        },
    }


def test_inline_json_references_rejects_recursive_schema():
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}},
            }
        },
    }

    with pytest.raises(
        ValueError, match="Recursive JSON schemas aren't supported"
    ):
        inline_local_json_refs(schema)


def test_to_mistral_tool_call_id_maps_invalid_values_stably():
    id_map: dict[str, str] = {}

    first = to_mistral_tool_call_id("call_1", id_map=id_map)
    second = to_mistral_tool_call_id("call_1", id_map=id_map)
    third = to_mistral_tool_call_id("another", id_map=id_map)

    assert first == "tc0000000"
    assert second == first
    assert third == "tc0000001"


def test_to_mistral_tool_call_id_keeps_valid_mistral_id():
    id_map: dict[str, str] = {}

    tool_call_id = to_mistral_tool_call_id("AbC123xY9", id_map=id_map)

    assert tool_call_id == "AbC123xY9"


async def test_to_attachment_chunks_allow_system_image_attachments():
    message = Message(
        role=Role.SYSTEM,
        content="system",
        custom_content=CustomContent(
            attachments=[
                Attachment(
                    type="image/png",
                    data=Resource(type="image/png", data=b"fake").data_base64,
                )
            ]
        ),
    )

    chunks = await MistralPromptParser._to_attachment_chunks(
        message, file_storage=None
    )

    assert len(chunks) == 1
    image_url = chunks[0].image_url
    url = image_url if isinstance(image_url, str) else image_url.url
    assert isinstance(url, str)
    assert url.startswith("data:image/png;base64,")


async def test_to_attachment_chunks_reject_non_image_types():
    message = Message(
        role=Role.USER,
        content="user",
        custom_content=CustomContent(
            attachments=[
                Attachment(
                    type="text/plain",
                    data=Resource(type="text/plain", data=b"abc").data_base64,
                )
            ]
        ),
    )

    with pytest.raises(
        ValidationError,
        match="Attachment of type 'text/plain' aren't supported",
    ):
        await MistralPromptParser._to_attachment_chunks(
            message, file_storage=None
        )


def test_tool_call_state_to_tool_call_parses_json_arguments():
    state = ToolCallState(index=3, id="id1234567", name="search")
    state.arguments = json.dumps({"q": "python"})

    tool_call = state.to_tool_call()

    assert tool_call.index == 3
    assert tool_call.id == "id1234567"
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == {"q": "python"}


def test_tool_call_state_to_tool_call_requires_name():
    state = ToolCallState(index=0, id="id1234567", name=None, arguments="{}")

    with pytest.raises(ValidationError, match="function name is missing"):
        state.to_tool_call()


def test_append_tool_calls_state_accumulates_streamed_arguments():
    state: dict[int, ToolCallState] = {}
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

    append_tool_calls_state(state, chunks)

    assert state[0].name == "search"
    assert state[0].arguments == '{"q":"python"}'


async def test_consume_stream_chunk_rewrites_tool_calls_finish_without_tools():
    chunk = CompletionChunk(
        id="chunk-id",
        model="mistral-model",
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=None, tool_calls=None),
                finish_reason="tool_calls",
            )
        ],
    )
    consumer, _ = _make_consumer_response()

    finish_reason = await consume_stream_chunk(
        chunk,
        consumer,
        tool_calls_state={},
        allow_tool_calls=False,
    )

    assert finish_reason == FinishReason.STOP
    assert consumer.get_finish_reason() == FinishReason.STOP


def test_to_finish_reason_maps_mistral_specific_values():
    assert to_finish_reason("length") == FinishReason.LENGTH
    assert to_finish_reason("model_length") == FinishReason.LENGTH
    assert to_finish_reason("tool_calls") == FinishReason.TOOL_CALLS
    assert to_finish_reason("error") == FinishReason.CONTENT_FILTER
    assert to_finish_reason("anything_else") == FinishReason.STOP


async def test_parse_tool_message_keeps_original_content():
    messages = [
        Message(role=Role.USER, content="hi"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                DialToolCall(
                    id="abc123xyz",
                    type="function",
                    function=DialFunctionCall(
                        name="search", arguments='{"q":"python"}'
                    ),
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_call_id="abc123xyz",
            content='{"result":"ok"}',
        ),
    ]
    params = ModelParameters.create(
        ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="hello")],
        )
    )

    prompt = await MistralPromptParser.parse(
        params=params,
        tools=ToolsConfig.noop(),
        file_storage=None,
        messages=messages,
    )

    tool_message = prompt.messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == '{"result":"ok"}'
    assert tool_message.name is None


async def test_chat_non_stream_emits_content_tool_calls_finish_reason_and_usage():
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5)
    response = SimpleNamespace(
        usage=usage,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="answer",
                    tool_calls=[
                        ToolCall(
                            id="AbC123xY9",
                            index=0,
                            function=FunctionCall(
                                name="search",
                                arguments='{"q":"python"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )
    adapter = _make_adapter(
        chat_client=_FakeMistralChat(complete_response=response)
    )
    consumer, response_obj = _make_consumer_response()

    with consumer:
        await adapter.chat(
            params=_make_model_params(stream=False),
            consumer=consumer,
            prompt=_make_prompt(tools_enabled=True),
        )

    assert consumer.get_finish_reason() == FinishReason.TOOL_CALLS
    assert consumer.get_usage().prompt_tokens == 3
    assert consumer.get_usage().completion_tokens == 5
    assert _extract_tool_calls(response_obj) == [
        ("AbC123xY9", "search", '{"q":"python"}')
    ]


async def test_chat_stream_emits_streamed_tool_calls_once_and_sets_usage():
    stream_chunks = [
        CompletionChunk(
            id="chunk-1",
            model="mistral-model",
            choices=[
                CompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="AbC123xY9",
                                index=0,
                                function=FunctionCall(
                                    name="search",
                                    arguments='{"q":"py',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        CompletionChunk(
            id="chunk-2",
            model="mistral-model",
            usage=UsageInfo(prompt_tokens=7, completion_tokens=11),
            choices=[
                CompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="AbC123xY9",
                                index=0,
                                function=FunctionCall(
                                    name="search",
                                    arguments='thon"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
        CompletionChunk(
            id="chunk-3",
            model="mistral-model",
            choices=[
                CompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=None, tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]
    adapter = _make_adapter(
        chat_client=_FakeMistralChat(stream_chunks=stream_chunks)
    )
    consumer, response_obj = _make_consumer_response()

    with consumer:
        await adapter.chat(
            params=_make_model_params(stream=True),
            consumer=consumer,
            prompt=_make_prompt(tools_enabled=True),
        )

    assert consumer.get_finish_reason() == FinishReason.TOOL_CALLS
    assert consumer.get_usage().prompt_tokens == 7
    assert consumer.get_usage().completion_tokens == 11
    assert _extract_tool_calls(response_obj) == [
        ("AbC123xY9", "search", '{"q":"python"}')
    ]


async def test_consume_tool_calls_function_api_keeps_only_first_call():
    consumer, response = _make_consumer_response()
    tool_calls = [
        ToolCall(
            id="AbC123xY9",
            index=0,
            function=FunctionCall(name="tool_a", arguments='{"a":1}'),
        ),
        ToolCall(
            id="Z9Y8X7W6V",
            index=1,
            function=FunctionCall(name="tool_b", arguments='{"b":2}'),
        ),
    ]

    with consumer:
        await consume_tool_calls(
            tool_calls,
            consumer,
            use_tool_api=False,
            allow_tool_calls=True,
        )

    assert _extract_function_calls(response) == [("tool_a", '{"a":1}')]


@pytest.mark.parametrize(
    ("tool_calls", "expected_calls"),
    [
        (
            [
                ToolCall(
                    id="AbC123xY9",
                    index=0,
                    function=FunctionCall(
                        name="get_temperature",
                        arguments='{"location":"London"}',
                    ),
                )
            ],
            [("AbC123xY9", "get_temperature", '{"location":"London"}')],
        ),
        (
            [
                ToolCall(
                    id="A1b2C3d4E",
                    index=0,
                    function=FunctionCall(name="tool_a", arguments='{"x":1}'),
                ),
                ToolCall(
                    id="F5g6H7i8J",
                    index=1,
                    function=FunctionCall(name="tool_b", arguments='{"y":2}'),
                ),
            ],
            [
                ("A1b2C3d4E", "tool_a", '{"x":1}'),
                ("F5g6H7i8J", "tool_b", '{"y":2}'),
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
                    id="Z9Y8X7W6V",
                    index=1,
                    function=FunctionCall(name="tool_b", arguments='{"b":2}'),
                ),
                ToolCall(
                    id="Q1w2E3r4T",
                    index=2,
                    function=FunctionCall(name="tool_c", arguments='{"c":3}'),
                ),
            ],
            [
                ("AbC123xY9", "tool_a", '{"a":1}'),
                ("Z9Y8X7W6V", "tool_b", '{"b":2}'),
                ("Q1w2E3r4T", "tool_c", '{"c":3}'),
            ],
        ),
    ],
)
async def test_consume_tool_calls_preserve_order(
    tool_calls: list[ToolCall],
    expected_calls: list[tuple[str, str, str | None]],
):
    consumer, response = _make_consumer_response()

    with consumer:
        await consume_tool_calls(
            tool_calls,
            consumer,
            use_tool_api=True,
            allow_tool_calls=True,
        )

    assert _extract_tool_calls(response) == expected_calls
