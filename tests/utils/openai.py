import json
import re
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Required,
    TypedDict,
    TypeVar,
    Unpack,
)

import httpx
from aidial_sdk.chat_completion.request import Attachment, Stage, StaticTool
from aidial_sdk.deployment.tokenize import TokenizeResponse
from aidial_sdk.utils.merge_chunks import (
    cleanup_indices,
    merge_chat_completion_chunks,
)
from openai import AsyncAzureOpenAI, AsyncStream
from openai._types import NOT_GIVEN
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as ToolFunction,
)
from openai.types.chat.completion_create_params import Function
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic.v1 import BaseModel

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.utils.resource import Resource
from tests.utils.json import match_objects


def sys(content: str) -> ChatCompletionSystemMessageParam:
    return {"role": "system", "content": content}


def ai(content: str) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "content": content}


def ai_function(
    function_call: ToolFunction,
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "function_call": function_call}


def ai_tools(
    tool_calls: List[ChatCompletionMessageToolCallParam],
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "tool_calls": tool_calls}


def user(
    content: str | List[ChatCompletionContentPartParam],
) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": content}


def user_with_attachment_data(
    content: str, *resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {"type": r.type, "data": r.data_base64} for r in resource
            ]
        },
    }


def user_with_attachment_url(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {
                    "type": resource.type,
                    "url": resource.to_data_url(),
                }
            ]
        },
    }


def user_with_image_url(
    content: str, image: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
                "type": "image_url",
                "image_url": {"url": image.to_data_url()},
            },
        ],
    }


def function_request(name: str, args: Any) -> ToolFunction:
    return {"name": name, "arguments": json.dumps(args)}


def tool_request(
    id: str, name: str, args: Any
) -> ChatCompletionMessageToolCallParam:
    return {
        "id": id,
        "type": "function",
        "function": function_request(name, args),
    }


def function_response(
    name: str, content: str
) -> ChatCompletionFunctionMessageParam:
    return {"role": "function", "name": name, "content": content}


def tool_response(id: str, content: str) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": id, "content": content}


def function_to_tool(function: FunctionDefinition) -> ChatCompletionToolParam:
    return {"type": "function", "function": function}


def sanitize_test_name(name: str) -> str:
    name = "".join(
        c if (c.isalnum() or c in "/:") else "_" for c in name.lower()
    )
    return re.sub("_+", "_", name)


_T = TypeVar("_T")


def foreach(f: Callable[[_T], None], xs: Iterable[_T]) -> None:
    for x in xs:
        f(x)


def assert_eq(a: Any, b: Any):
    assert a == b


class ChatCompletionResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    response: ChatCompletion

    @property
    def message(self) -> ChatCompletionMessage:
        return self.response.choices[0].message

    @property
    def content(self) -> str:
        return self.message.content or ""

    @property
    def contents(self) -> List[str]:
        return [
            choice.message.content or "" for choice in self.response.choices
        ]

    @property
    def finish_reasons(self) -> List[str]:
        return [choice.finish_reason for choice in self.response.choices]

    @property
    def usage(self) -> CompletionUsage | None:
        return self.response.usage

    @property
    def function_call(self) -> FunctionCall | None:
        return self.message.function_call

    @property
    def tool_calls(self) -> List[ChatCompletionMessageToolCall] | None:
        return self.message.tool_calls

    def content_contains_all(self, matches: List[Any]) -> None:
        for match in matches:
            assert str(match).lower() in self.content.lower()

    @property
    def attachments(self) -> List[Attachment] | None:
        if not hasattr(self.message, "custom_content"):
            return None
        return [
            Attachment.parse_obj(attachment)
            for attachment in self.message.custom_content.get(  # type: ignore
                "attachments", []
            )
        ] or None

    @property
    def stages(self) -> List[Stage] | None:
        if not hasattr(self.message, "custom_content"):
            return None
        return [
            Stage.parse_obj(stage)
            for stage in self.message.custom_content.get(  # type: ignore
                "stages", []
            )
        ] or None


async def tokenize_request(
    http_client: httpx.AsyncClient,
    model_id: str,
    messages: List[ChatCompletionMessageParam],
    functions: List[Function] | None,
    tools: List[ChatCompletionToolParam] | None,
    extra_headers: Mapping[str, str] | None = None,
) -> TokenizeResponse:

    chat_completion_request = {
        "model": model_id,
        "messages": messages,
        "tools": tools,
        "functions": functions,
    }

    request = {
        "inputs": [{"type": "request", "value": chat_completion_request}],
    }

    resp = await http_client.post(
        f"openai/deployments/{model_id}/tokenize",
        json=request,
        headers=extra_headers or {},
    )

    resp.raise_for_status()

    return TokenizeResponse.parse_obj(resp.json())


class ChatCompletionArgs(TypedDict, total=False):
    messages: Required[List[ChatCompletionMessageParam]]
    stop: List[str] | None
    max_tokens: int | None
    n: int | None
    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None
    static_tools: StaticToolsConfig | None
    extra_body: dict | None


async def chat_completion(
    client: AsyncAzureOpenAI,
    *,
    stream: bool | None = None,
    **kwargs: Unpack[ChatCompletionArgs],
) -> ChatCompletionResult:

    merged_tools = (
        [
            StaticTool(
                type="static_function",
                static_function=function,
            ).dict()
            for function in (static_tools.functions or [])
        ]
        if (static_tools := kwargs.get("static_tools"))
        else []
    )
    if tools := kwargs.get("tools"):
        merged_tools += tools

    extra_body = kwargs.get("extra_body") or {}
    if merged_tools:
        extra_body["tools"] = merged_tools

    async def get_response() -> ChatCompletion:
        functions = kwargs.get("functions")
        tools = kwargs.get("tools")

        response = await client.chat.completions.create(
            model="dummy-model",
            messages=kwargs["messages"],
            stream=stream or False,
            stop=kwargs.get("stop"),
            max_tokens=kwargs.get("max_tokens"),
            temperature=0.0,
            n=kwargs.get("n"),
            function_call="auto" if functions is not None else NOT_GIVEN,
            functions=functions or NOT_GIVEN,
            tool_choice="auto" if tools is not None else NOT_GIVEN,
            tools=tools or NOT_GIVEN,
            # Using extra_body to override tools, since openai
            # doesn't support static tools
            extra_body=extra_body,
        )

        if isinstance(response, AsyncStream):
            chunks: List[dict] = []
            async for chunk in response:
                chunks.append(chunk.model_dump())

            response_dict = merge_chat_completion_chunks(*chunks)

            for choice in response_dict["choices"]:
                choice["message"] = cleanup_indices(choice["delta"])
                del choice["delta"]

            response_dict["object"] = "chat.completion"

            return ChatCompletion.model_validate(response_dict)
        else:
            return response

    response = await get_response()
    return ChatCompletionResult(response=response)


GET_WEATHER_FUNCTION: Function = {
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location.",
            },
        },
        "required": ["location", "format"],
    },
}

GET_WEATHER_TOOL: ChatCompletionToolParam = function_to_tool(
    GET_WEATHER_FUNCTION
)


def is_valid_function_call(
    call: FunctionCall | None, expected_name: str, expected_args: Any
):
    assert call is not None, "Function call is missing"
    assert call.name == expected_name
    obj = json.loads(call.arguments)
    match_objects(expected_args, obj)


def is_valid_tool_call(
    calls: List[ChatCompletionMessageToolCall] | None,
    tool_call_idx: int,
    check_tool_id: Callable[[str], None],
    expected_name: str,
    expected_args: dict,
):
    assert calls is not None, "Tool calls are missing"
    assert tool_call_idx < len(calls), f"Tool call #{tool_call_idx} is missing"

    call = calls[tool_call_idx]

    function = call.function
    check_tool_id(call.id)
    assert expected_name == function.name

    actual_args = json.loads(function.arguments)
    match_objects(expected_args, actual_args)
