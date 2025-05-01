from dataclasses import dataclass
from typing import List

import httpx
import pytest
from aidial_sdk.deployment.tokenize import TokenizeError, TokenizeSuccess
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.integration_tests.constants import BLUE_PNG_PICTURE
from tests.utils.dial import get_extra_headers
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    GET_WEATHER_TOOL,
    ai,
    ai_function,
    ai_tools,
    chat_completion,
    function_request,
    function_response,
    sanitize_test_name,
    sys,
    tokenize_request,
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
    expected_error: str | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    def get_id(self):
        return sanitize_test_name(f"{self.deployment.value}/{self.name}")


chat_deployments = [
    ChatCompletionDeployment.GEMINI_PRO_VISION_1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V1,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
]


def supports_tools(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


def is_text_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


def get_test_cases(deployment: ChatCompletionDeployment) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        error: str | None = None,
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                deployment,
                messages,
                error,
                functions,
                tools,
            )
        )

    text_model = is_text_model(deployment)

    test_case(
        name="single user message",
        messages=[user("user")],
        error=None if text_model else "No documents were found",
    )

    test_case(
        name="empty sys message + user",
        messages=[sys(""), user("user")],
        error=None if text_model else "No documents were found",
    )

    test_case(
        name="non-empty sys message + user",
        messages=[sys("system"), user("user")],
        error=None if text_model else "No documents were found",
    )

    test_case(
        name="sys message",
        messages=[sys("system")],
        error=(
            "contents are required."
            if text_model
            else "No documents were found"
        ),
    )

    for idx, user_message in enumerate(
        [
            user_with_attachment_data("user", BLUE_PNG_PICTURE),
            user_with_attachment_url("user", BLUE_PNG_PICTURE),
            user_with_image_url("user", BLUE_PNG_PICTURE),
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
        )

        function_req = ai_function(function_request(name, function_args))
        function_resp = function_response(name, "15 celsius")

        test_case(
            name="weather function followup",
            messages=[user(content), function_req, function_resp],
            functions=[GET_WEATHER_FUNCTION],
        )

        # Tools
        tool_call_id = f"{name}_1"
        test_case(
            name="weather tool",
            messages=[user(content)],
            tools=[GET_WEATHER_TOOL],
        )

        tool_req = ai_tools([tool_request(tool_call_id, name, function_args)])
        tool_resp = tool_response(tool_call_id, "15 celsius")

        test_case(
            name="weather tool followup",
            messages=[user(content), tool_req, tool_resp],
            tools=[GET_WEATHER_TOOL],
        )

    return test_cases


@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment in chat_deployments
        for test in get_test_cases(deployment)
    ],
    ids=TestCase.get_id,
)
async def test_tokenize(
    get_openai_client, test_http_client: httpx.AsyncClient, test: TestCase
):
    extra_headers = get_extra_headers("us-central1")
    deployment_id = test.deployment.value

    actual_output = await tokenize_request(
        test_http_client,
        deployment_id,
        test.messages,
        test.functions,
        test.tools,
        extra_headers=extra_headers,
    )

    outputs = actual_output.outputs
    assert len(outputs) == 1
    output = outputs[0]

    if isinstance(test.expected_error, str):
        assert isinstance(output, TokenizeError)
        assert output.status == "error"
        assert output.error == test.expected_error
    else:

        chat_completion_response = await chat_completion(
            client=get_openai_client(deployment_id, extra_headers),
            messages=test.messages,
            stream=False,
            max_tokens=1,
            functions=test.functions,
            tools=test.tools,
        )

        assert isinstance(output, TokenizeSuccess)
        assert output.status == "success"
        usage = chat_completion_response.usage
        assert usage is not None, "Usage is missing"
        assert output.token_count == usage.prompt_tokens
