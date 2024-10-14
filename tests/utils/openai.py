import json
import re
from typing import Any, AsyncGenerator, Callable, List, Optional

from aidial_sdk.utils.streaming import merge_chunks
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
from pydantic import BaseModel

from aidial_adapter_vertexai.utils.resource import Resource


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
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {"type": resource.type, "data": resource.data_base64}
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
    name2 = "".join(c if c.isalnum() else "_" for c in name.lower())
    return re.sub("_+", "_", name2)


class ChatCompletionResult(BaseModel):
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
    def usage(self) -> CompletionUsage | None:
        return self.response.usage

    @property
    def function_call(self) -> FunctionCall | None:
        return self.message.function_call

    @property
    def tool_calls(self) -> List[ChatCompletionMessageToolCall] | None:
        return self.message.tool_calls


async def chat_completion(
    client: AsyncAzureOpenAI,
    messages: List[ChatCompletionMessageParam],
    stream: bool,
    stop: Optional[List[str]],
    max_tokens: Optional[int],
    n: Optional[int],
    functions: List[Function] | None,
    tools: List[ChatCompletionToolParam] | None,
) -> ChatCompletionResult:
    async def get_response() -> ChatCompletion:
        response = await client.chat.completions.create(
            model="dummy_model",
            messages=messages,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            temperature=0.0,
            n=n,
            function_call="auto" if functions is not None else NOT_GIVEN,
            functions=functions or NOT_GIVEN,
            tool_choice="auto" if tools is not None else NOT_GIVEN,
            tools=tools or NOT_GIVEN,
        )

        if isinstance(response, AsyncStream):

            async def generator() -> AsyncGenerator[dict, None]:
                async for chunk in response:
                    yield chunk.dict()

            response_dict = await merge_chunks(generator())
            response_dict["object"] = "chat.completion"
            response_dict["model"] = "dummy_model"

            return ChatCompletion.parse_obj(response_dict)
        else:
            return response

    response = await get_response()
    return ChatCompletionResult(response=response)


def for_all_choices(
    predicate: Callable[[str], bool], n: int = 1
) -> Callable[[ChatCompletionResult], bool]:
    def f(resp: ChatCompletionResult) -> bool:
        contents = resp.contents
        assert (
            len(contents) == n
        ), f"Expected {n} candidates, got {len(contents)}"
        return all(predicate(content) for content in contents)

    return f


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
