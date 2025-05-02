from dataclasses import dataclass
from typing import Callable, List

import httpx
import pytest
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
    streaming: bool

    messages: List[ChatCompletionMessageParam]

    max_tokens: int | None
    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    check: Callable[[int, int], None]

    def get_id(self):
        max_tokens_str = f"maxt:{self.max_tokens}" if self.max_tokens else None
        return sanitize_test_name(
            "/".join(
                str(part)
                for part in [
                    self.deployment.value,
                    self.streaming,
                    max_tokens_str,
                    self.name,
                ]
                if part is not None
            )
        )


chat_deployments = [
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_OPUS,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET,
    ChatCompletionDeployment.CLAUDE_3_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_7_SONNET,
]


def supports_tools(deployment: ChatCompletionDeployment) -> bool:
    return True


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.CLAUDE_3_5_HAIKU


def _eq_check(actual: int, expected: int):
    assert actual == expected


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        max_tokens: int | None = None,
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
        check: Callable[[int, int], None] = _eq_check,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                deployment,
                streaming,
                messages,
                max_tokens,
                functions,
                tools,
                check,
            )
        )

    test_case(
        name="single user message",
        messages=[user("user")],
        max_tokens=1,
    )

    test_case(
        name="empty sys message + user",
        messages=[sys(""), user("user")],
        max_tokens=1,
    )

    test_case(
        name="non-empty sys message + user",
        messages=[sys("system"), user("user")],
        max_tokens=1,
    )

    test_case(
        name="long completion",
        messages=[user("tell me the full story of Pinocchio")],
        max_tokens=1,
    )

    if is_vision_model(deployment):
        content = "user"

        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, BLUE_PNG_PICTURE),
                user_with_attachment_url(content, BLUE_PNG_PICTURE),
                user_with_image_url(content, BLUE_PNG_PICTURE),
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
                max_tokens=1,
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
        for streaming in [False, True]
        for test in get_test_cases(deployment, streaming)
    ],
    ids=TestCase.get_id,
)
async def test_tokenize(
    test_http_client: httpx.AsyncClient, get_openai_client, test: TestCase
):
    region = "us-east5"
    extra_headers = get_extra_headers(region)
    client = get_openai_client(test.deployment.value, extra_headers)

    response = await chat_completion(
        client=client,
        messages=test.messages,
        stream=test.streaming,
        stop=[],
        max_tokens=test.max_tokens,
        n=1,
        functions=test.functions,
        tools=test.tools,
        static_tools=None,
    )

    usage = response.usage
    assert usage is not None, "Usage is missing"

    expected_prompt_tokens = usage.prompt_tokens

    resp = await tokenize_request(
        test_http_client,
        test.deployment.value,
        test.messages,
        test.functions,
        test.tools,
        extra_headers=extra_headers,
    )

    output = resp.outputs[0]
    assert output.status == "success"
    actual_prompt_tokens = output.token_count
    test.check(actual_prompt_tokens, expected_prompt_tokens)
