from dataclasses import dataclass
from typing import Callable, List

import httpx
import pytest
from aidial_sdk.deployment.tokenize import TokenizeResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.utils.resource import Resource
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    GET_WEATHER_TOOL,
    ai,
    ai_function,
    ai_tools,
    check_tokenize_response,
    function_request,
    function_response,
    sanitize_test_name,
    sys,
    tokenize,
    tool_request,
    tool_response,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment

    messages: List[ChatCompletionMessageParam]
    expected: Callable[[TokenizeResponse], bool]

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    def get_id(self):
        return sanitize_test_name(f"{self.deployment.value} {self.name}")


deployments = [
    ChatCompletionDeployment.GEMINI_PRO_1,
    ChatCompletionDeployment.GEMINI_PRO_VISION_1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V1,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
]


def supports_tools(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_1


def is_text_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


blue_pic = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)

# https://ai.google.dev/gemini-api/docs/tokens?lang=python#images
GEMINI_TOKENS_PER_IMAGE = 258


def get_test_cases(deployment: ChatCompletionDeployment) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: Callable[[TokenizeResponse], bool],
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                deployment,
                messages,
                expected,
                functions,
                tools,
            )
        )

    text_model = is_text_model(deployment)

    test_case(
        name="single user message",
        messages=[user("user")],
        expected=check_tokenize_response(
            1 if text_model else "No documents were found"
        ),
    )

    test_case(
        name="empty sys message + user",
        messages=[sys(""), user("user")],
        expected=check_tokenize_response(
            1 if text_model else "No documents were found"
        ),
    )

    test_case(
        name="non-empty sys message + user",
        messages=[sys("system"), user("user")],
        expected=check_tokenize_response(
            2 if text_model else "No documents were found"
        ),
    )

    test_case(
        name="sys message",
        messages=[sys("system")],
        expected=check_tokenize_response(
            "contents must not be empty"
            if text_model
            else "No documents were found"
        ),
    )

    if is_vision_model(deployment):
        content = "user"

        # Gemini Vision cuts the dialog down to the last message
        non_image_tokens = (
            1
            if deployment == ChatCompletionDeployment.GEMINI_PRO_VISION_1
            else 4
        )

        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, blue_pic),
                user_with_attachment_url(content, blue_pic),
                user_with_image_url(content, blue_pic),
            ]
        ):
            test_case(
                name=f"describe image {idx}",
                messages=[
                    sys("system"),
                    user("ping"),
                    ai("pong"),
                    user_message,
                ],
                expected=check_tokenize_response(
                    non_image_tokens + GEMINI_TOKENS_PER_IMAGE
                ),
            )

    if supports_tools(deployment):
        content = "What's the temperature in Glasgow in celsius?"

        function_args = {"location": "Glasgow", "format": "celsius"}

        name = GET_WEATHER_FUNCTION["name"]

        # Functions
        test_case(
            name="weather function",
            messages=[user(content)],
            functions=[GET_WEATHER_FUNCTION],
            expected=check_tokenize_response(53),
        )

        function_req = ai_function(function_request(name, function_args))
        function_resp = function_response(name, "15 celsius")

        test_case(
            name="weather function followup",
            messages=[user(content), function_req, function_resp],
            functions=[GET_WEATHER_FUNCTION],
            expected=check_tokenize_response(72),
        )

        # Tools
        tool_call_id = f"{name}_1"
        test_case(
            name="weather tool",
            messages=[user(content)],
            tools=[GET_WEATHER_TOOL],
            expected=check_tokenize_response(53),
        )

        tool_req = ai_tools([tool_request(tool_call_id, name, function_args)])
        tool_resp = tool_response(tool_call_id, "15 celsius")

        test_case(
            name="weather tool followup",
            messages=[user(content), tool_req, tool_resp],
            tools=[GET_WEATHER_TOOL],
            expected=check_tokenize_response(72),
        )

    return test_cases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [test for deployment in deployments for test in get_test_cases(deployment)],
    ids=TestCase.get_id,
)
async def test_tokenize(test_http_client: httpx.AsyncClient, test: TestCase):

    actual_output = await tokenize(
        test_http_client,
        test.deployment.value,
        test.messages,
        test.functions,
        test.tools,
    )

    assert test.expected(
        actual_output
    ), f"Failed output test, actual output: {actual_output}"
