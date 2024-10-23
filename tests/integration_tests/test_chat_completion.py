import json
import re
from dataclasses import dataclass
from typing import Any, Callable, List

import pytest
from openai import APIError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.utils.json import match_objects
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    GET_WEATHER_TOOL,
    ChatCompletionResult,
    ai_function,
    ai_tools,
    blue_pic,
    chat_completion,
    for_all_choices,
    function_request,
    function_response,
    sanitize_test_name,
    sys,
    tool_request,
    tool_response,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)


def is_valid_function_call(
    call: FunctionCall | None, expected_name: str, expected_args: Any
) -> bool:
    assert call is not None
    assert call.name == expected_name
    obj = json.loads(call.arguments)
    match_objects(expected_args, obj)
    return True


def is_valid_tool_calls(
    calls: List[ChatCompletionMessageToolCall] | None,
    expected_id: str,
    expected_name: str,
    expected_args: Any,
) -> bool:
    assert calls is not None
    assert len(calls) == 1
    call = calls[0]

    function_call = call.function
    assert call.id == expected_id
    assert function_call.name == expected_name

    obj = json.loads(function_call.arguments)
    match_objects(expected_args, obj)
    return True


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str
    status_code: int | None = None


def expected_success(*args, **kwargs):
    return True


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment
    streaming: bool

    messages: List[ChatCompletionMessageParam]
    expected: Callable[[ChatCompletionResult], bool] | ExpectedException

    max_tokens: int | None
    stop: List[str] | None

    n: int | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    def get_id(self):
        max_tokens_str = f"maxt={self.max_tokens}" if self.max_tokens else ""
        stop_sequence_str = f"stop={self.stop}" if self.stop else ""
        n_str = f"n={self.n}" if self.n else ""
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {max_tokens_str} "
            f"{stop_sequence_str} {n_str} {self.name}"
        )


deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CHAT_BISON_2_32K,
    ChatCompletionDeployment.CODECHAT_BISON_1,
    ChatCompletionDeployment.GEMINI_PRO_1,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
    ChatCompletionDeployment.GEMINI_PRO_VISION_1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
]


def is_codechat(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.CODECHAT_BISON_1,
        ChatCompletionDeployment.CODECHAT_BISON_2,
        ChatCompletionDeployment.CODECHAT_BISON_2_32K,
    ]


def supports_tools(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_PRO_1,
        ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
    ]


def supports_text_input(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_PRO_VISION_1,
        ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
        ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
    ]


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            Callable[[ChatCompletionResult], bool] | ExpectedException
        ) = expected_success,
        n: int | None = None,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                deployment,
                streaming,
                messages,
                expected,
                max_tokens,
                stop,
                n,
                functions,
                tools,
            )
        )

    if supports_text_input(deployment):
        test_case(
            name="2+3=5",
            messages=[user("2+3=?")],
            expected=for_all_choices(lambda s: "5" in s),
        )

        test_case(
            name="hello",
            messages=[user('Reply with "Hello"')],
            expected=for_all_choices(lambda s: "hello" in s.lower()),
        )

        test_case(
            name="empty sys message",
            messages=[sys(""), user("2+4=?")],
            expected=for_all_choices(lambda s: "6" in s),
        )

        test_case(
            name="non empty sys message",
            messages=[sys("Act as helpful assistant"), user("2+5=?")],
            expected=for_all_choices(lambda s: "7" in s),
        )

        test_case(
            name="max tokens 1",
            max_tokens=1,
            messages=[user("tell me the full story of Pinocchio")],
            expected=for_all_choices(lambda s: len(s.split()) == 1),
        )

        test_case(
            name="multiple candidates",
            max_tokens=10,
            n=5,
            messages=[user("2+3=?")],
            expected=(
                ExpectedException(
                    type=UnprocessableEntityError,
                    message="n>1 is not supported in streaming mode",
                    status_code=422,
                )
                if streaming
                else for_all_choices(lambda _: True, 5)
            ),
        )

        # Stop sequences do not work for some reason for CHAT_BISON_2_32K and streaming mode
        if (deployment, streaming) != (
            ChatCompletionDeployment.CHAT_BISON_2_32K,
            True,
        ):
            test_case(
                name="stop sequence",
                max_tokens=None,
                stop=["world"],
                messages=[user('Reply with "hello world"')],
                expected=(
                    ExpectedException(
                        type=UnprocessableEntityError,
                        message="stop sequences are not supported for code chat model",
                        status_code=422,
                    )
                    if is_codechat(deployment)
                    else for_all_choices(lambda s: "world" not in s.lower())
                ),
            )

    if is_vision_model(deployment):
        content = "describe the image"
        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, blue_pic),
                user_with_attachment_url(content, blue_pic),
                user_with_image_url(content, blue_pic),
            ]
        ):
            test_case(
                name=f"describe image {idx}",
                max_tokens=100,
                messages=[sys("be a helpful assistant"), user_message],
                expected=lambda s: "blue" in s.content.lower(),
            )

    if supports_tools(deployment):
        content = "What's the temperature in Glasgow in celsius?"

        function_args_checker = {
            "location": lambda s: "glasgow" in s.lower(),
            "format": "celsius",
        }

        function_args = {"location": "Glasgow", "format": "celsius"}

        name = GET_WEATHER_FUNCTION["name"]

        # Functions
        test_case(
            name="weather function",
            messages=[user(content)],
            functions=[GET_WEATHER_FUNCTION],
            expected=lambda s: is_valid_function_call(
                s.function_call, name, function_args_checker
            ),
        )

        function_req = ai_function(function_request(name, function_args))
        function_resp = function_response(name, "15 celsius")

        test_case(
            name="weather function followup",
            messages=[user(content), function_req, function_resp],
            functions=[GET_WEATHER_FUNCTION],
            expected=lambda s: "15" in s.content.lower(),
        )

        # Tools
        tool_call_id = f"{name}_1"
        test_case(
            name="weather tool",
            messages=[user(content)],
            tools=[GET_WEATHER_TOOL],
            expected=lambda s: is_valid_tool_calls(
                s.tool_calls, tool_call_id, name, function_args_checker
            ),
        )

        tool_req = ai_tools([tool_request(tool_call_id, name, function_args)])
        tool_resp = tool_response(tool_call_id, "15 celsius")

        test_case(
            name="weather tool followup",
            messages=[user(content), tool_req, tool_resp],
            tools=[GET_WEATHER_TOOL],
            expected=lambda s: "15" in s.content.lower(),
        )

    return test_cases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment in deployments
        for streaming in [False, True]
        for test in get_test_cases(deployment, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_openai(get_openai_client, test: TestCase):
    client = get_openai_client(test.deployment.value)

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(
            client,
            test.messages,
            test.streaming,
            test.stop,
            test.max_tokens,
            test.n,
            test.functions,
            test.tools,
        )

    if isinstance(test.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        assert isinstance(actual_exc, test.expected.type)
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test.expected.status_code
        assert re.search(test.expected.message, str(actual_exc))
    else:
        actual_output = await run_chat_completion()
        assert test.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
