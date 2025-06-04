from dataclasses import dataclass
from typing import List, Mapping

import httpx
import pytest
from aidial_sdk.deployment.tokenize import TokenizeError, TokenizeSuccess
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import get_extra_headers
from tests.integration_tests.constants import BLUE_PNG_PICTURE
from tests.integration_tests.test_chat_completion_generation import (
    is_vision_model,
    supports_tools,
)
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
    region: str

    messages: List[ChatCompletionMessageParam]
    expected_error: str | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    def get_id(self):
        return sanitize_test_name(f"{self.deployment.value}/{self.name}")


_CENTRAL = "us-central1"
_EAST = "us-east5"

chat_deployments: Mapping[ChatCompletionDeployment, str] = {
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2: _CENTRAL,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2: _CENTRAL,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_LITE_1: _CENTRAL,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2: _EAST,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU: _EAST,
    ChatCompletionDeployment.CLAUDE_3_OPUS: _EAST,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_3_HAIKU: _EAST,
    ChatCompletionDeployment.CLAUDE_3_7_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_4_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_4_OPUS: _EAST,
}

_tolerance: Mapping[ChatCompletionDeployment, int] = {
    # For some reason reported tokens for Claude 4 are off by one
    ChatCompletionDeployment.CLAUDE_4_SONNET: 1,
    ChatCompletionDeployment.CLAUDE_4_OPUS: 1,
}


def is_gemini(deployment: ChatCompletionDeployment) -> bool:
    return "gemini" in deployment.value


def get_test_cases(
    deployment: ChatCompletionDeployment, region: str
) -> List[TestCase]:
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
                name, deployment, region, messages, error, functions, tools
            )
        )

    test_case(
        name="single user message",
        messages=[user("user")],
    )

    test_case(
        name="empty sys message + user",
        messages=[sys(""), user("user")],
    )

    test_case(
        name="non-empty sys message + user",
        messages=[sys("system"), user("user")],
    )

    test_case(
        name="long completion",
        messages=[user("tell me the full story of Pinocchio")],
    )

    test_case(
        name="sys message",
        messages=[sys("system")],
        error=(
            "contents are required."
            if is_gemini(deployment)
            else "messages: at least one message is required"
        ),
    )

    if is_vision_model(deployment):
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
        for deployment, region in chat_deployments.items()
        for test in get_test_cases(deployment, region)
    ],
    ids=TestCase.get_id,
)
async def test_tokenize(
    get_openai_client, test_http_client: httpx.AsyncClient, test: TestCase
):
    deployment_id = test.deployment.value

    actual_output = await tokenize_request(
        test_http_client,
        deployment_id,
        test.messages,
        test.functions,
        test.tools,
        extra_headers=get_extra_headers(test.region),
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
            client=get_openai_client(deployment_id, region=test.region),
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
        _tolerance_value = _tolerance.get(test.deployment, 0)
        assert abs(output.token_count - usage.prompt_tokens) <= _tolerance_value
