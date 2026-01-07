from __future__ import annotations

import dataclasses
import json
import os
from typing import Awaitable, Callable, Dict, List, Protocol, Unpack
from unittest.mock import patch

import anthropic
import openai
import pytest
from aidial_sdk.chat_completion.request import Message as DialMessage
from aidial_sdk.chat_completion.request import StaticFunction
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_vertexai.chat.gemini.state import (
    _parse_message_content_from_state,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.conftest import get_extra_headers
from tests.integration_tests.constants import DOG_PICTURE, DOG_PICTURE_CONTENT
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.json import match_objects
from tests.utils.openai import (
    GET_CURRENT_TIME_FUNCTION,
    GET_WEATHER_FUNCTION,
    GET_WEATHER_TOOL_WITH_REFERENCES,
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    ai_tools,
    chat_completion,
    function_to_tool,
    sanitize_test_name,
    sys,
    tool_response,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)
from tests.utils.selector import Selector, pred
from tests.utils.tools import ToolCallTest

DeploymentConf = Dict[str, str]


@dataclasses.dataclass
class compat:
    upstream: str
    deployment: D

    def to_compatibility_mapping(self) -> dict[str, str]:
        return {self.upstream: self.deployment.value}


@dataclasses.dataclass
class supported:
    deployment: D

    @property
    def upstream(self) -> str:
        return self.deployment.value


@dataclasses.dataclass
class vertexai:
    region: str

    def headers(self) -> DeploymentConf:
        return get_extra_headers(self.region)


@dataclasses.dataclass
class foundry:
    service_name: str

    @classmethod
    def create(cls) -> foundry | None:
        if name := os.getenv("INTEGRATION_TEST_AZURE_FOUNDRY_SERVICE_NAME"):
            return cls(service_name=name)
        return None

    def headers(self) -> DeploymentConf:
        return {
            "x-upstream-endpoint": f"https://{self.service_name}.services.ai.azure.com/anthropic/v1/messages"
        }


@dataclasses.dataclass
class deployment_entry:
    model: supported | compat
    source: vertexai | foundry

    @property
    def deployment(self) -> D:
        return self.model.deployment

    @property
    def upstream(self) -> str:
        return self.model.upstream

    def display(self) -> str:
        ret = ""
        if isinstance(self.model, compat):
            ret += f"{sanitize_test_name(self.model.upstream)}-compat"
        else:
            ret += sanitize_test_name(self.model.deployment.value)
        ret += "/"
        if isinstance(self.source, vertexai):
            ret += self.source.region
        else:
            ret += "foundry"
        return ret


_CENTRAL = "us-central1"
_EAST = "us-east5"
_GLOBAL = "global"


def supported_vertexai(deployment: D, region: str) -> deployment_entry:
    return deployment_entry(supported(deployment), vertexai(region))


_DEPLOYMENTS: List[deployment_entry] = [
    supported_vertexai(D.GEMINI_2_0_FLASH_EXP, _CENTRAL),
    supported_vertexai(D.GEMINI_2_0_FLASH_001, _CENTRAL),
    supported_vertexai(D.GEMINI_2_5_PRO, _CENTRAL),
    supported_vertexai(D.GEMINI_2_5_PRO_PREVIEW_03_25, _CENTRAL),
    supported_vertexai(D.GEMINI_3_PRO_PREVIEW, _GLOBAL),
    supported_vertexai(D.GEMINI_2_0_FLASH_LITE_1, _CENTRAL),
    supported_vertexai(D.GEMINI_2_5_FLASH, _CENTRAL),
    supported_vertexai(D.GEMINI_2_5_FLASH_IMAGE_PREVIEW, _GLOBAL),
    supported_vertexai(D.GEMINI_3_PRO_IMAGE_PREVIEW, _GLOBAL),
    supported_vertexai(D.GEMINI_2_5_FLASH_IMAGE, _GLOBAL),
    supported_vertexai(D.CLAUDE_3_5_SONNET_V2, _EAST),
    supported_vertexai(D.CLAUDE_3_5_HAIKU, _EAST),
    supported_vertexai(D.CLAUDE_3_OPUS, _EAST),
    supported_vertexai(D.CLAUDE_3_5_SONNET, _EAST),
    supported_vertexai(D.CLAUDE_3_HAIKU, _EAST),
    supported_vertexai(D.CLAUDE_3_7_SONNET, _EAST),
    supported_vertexai(D.CLAUDE_4_SONNET, _EAST),
    supported_vertexai(D.CLAUDE_4_OPUS, _EAST),
    supported_vertexai(D.CLAUDE_4_1_OPUS, _EAST),
    supported_vertexai(D.CLAUDE_4_5_HAIKU, _EAST),
    supported_vertexai(D.CLAUDE_4_5_SONNET, _EAST),
]

if conf := foundry.create():
    source = compat("claude-sonnet-4-520250929", D.CLAUDE_4_5_SONNET)
    _DEPLOYMENTS.append(deployment_entry(source, conf))

    @pytest.fixture()
    def compatibility_mapping():
        return source.to_compatibility_mapping()


def is_retired_model(deployment: D) -> bool:
    # Keep at least one model on the list to test how the adapter handles retired models in streaming and non-streaming modes
    # Find the list of retired models at
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#retired-models
    return deployment in {
        D.GEMINI_2_5_PRO_PREVIEW_03_25,
    }


def is_vision_model(deployment: D) -> bool:
    return deployment in [
        D.GEMINI_2_5_FLASH,
        D.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
        D.GEMINI_3_PRO_IMAGE_PREVIEW,
        D.GEMINI_2_5_FLASH_IMAGE,
        D.GEMINI_2_5_PRO,
        D.GEMINI_3_PRO_PREVIEW,
        D.GEMINI_2_0_FLASH_EXP,
        D.GEMINI_2_0_FLASH_001,
        D.CLAUDE_3_5_SONNET_V2,
        # Upstream returns 'claude-3-5-haiku-20241022 does not support images.'
        # D.CLAUDE_3_5_HAIKU,
        # This model hallucinates on a the test image
        # D.CLAUDE_3_OPUS,
        D.CLAUDE_3_5_SONNET,
        D.CLAUDE_3_HAIKU,
        D.CLAUDE_3_7_SONNET,
        D.CLAUDE_4_OPUS,
        D.CLAUDE_4_SONNET,
        D.CLAUDE_4_1_OPUS,
        D.CLAUDE_4_5_HAIKU,
        D.CLAUDE_4_5_SONNET,
    ]


def select(
    p: Selector[D], xs: List[deployment_entry]
) -> List[deployment_entry]:
    return [x for x in xs if p(x.deployment)]


deployments = select(~pred(is_retired_model), _DEPLOYMENTS)
retired_deployments = select(pred(is_retired_model), _DEPLOYMENTS)
vision_deployments = select(pred(is_vision_model), deployments)
sample_deployment = select(
    pred(lambda d: d == D.CLAUDE_3_7_SONNET), deployments
)


def supports_json_object_response_format(deployment: D) -> bool:
    return deployment in [
        D.GEMINI_2_0_FLASH_EXP,
        D.GEMINI_2_0_FLASH_001,
    ]


def supports_json_schema_response_format(deployment: D) -> bool:
    return supports_json_object_response_format(deployment)


def is_claude(deployment: D) -> bool:
    return "claude" in deployment.value


def supports_tools(deployment: D) -> bool:
    return is_claude(deployment) or deployment in [
        D.GEMINI_2_0_FLASH_EXP,
        D.GEMINI_2_0_FLASH_001,
        D.GEMINI_2_5_PRO,
        D.GEMINI_2_0_FLASH_LITE_1,
        D.GEMINI_2_5_FLASH,
        D.GEMINI_3_PRO_PREVIEW,
    ]


def supports_parallel_tool_calls(deployment: D) -> bool:
    return deployment in [
        # D.CLAUDE_3_5_SONNET_V2,
        # D.CLAUDE_3_HAIKU,
        D.CLAUDE_3_5_HAIKU,
        D.CLAUDE_3_OPUS,
        D.CLAUDE_3_5_SONNET,
        # D.CLAUDE_3_7_SONNET,
        D.GEMINI_2_5_PRO,
        D.GEMINI_2_0_FLASH_LITE_1,
        D.GEMINI_2_5_FLASH,
        D.GEMINI_3_PRO_PREVIEW,
    ]


def supports_tool_call_ids(deployment: D) -> bool:
    return is_claude(deployment)


def is_gemini_image(deployment: D) -> bool:
    return deployment in (
        D.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
        D.GEMINI_2_5_FLASH_IMAGE,
        D.GEMINI_3_PRO_IMAGE_PREVIEW,
    )


def supports_grounding(deployment: D) -> bool:
    return is_gemini(deployment) and not is_gemini_image(deployment)


def supports_thinking(deployment: D) -> bool:
    return deployment in (
        D.GEMINI_2_5_PRO,
        D.GEMINI_2_5_FLASH,
        D.GEMINI_3_PRO_PREVIEW,
    )


def supports_thinking_level(deployment: D) -> bool:
    return deployment in (D.GEMINI_3_PRO_PREVIEW,)


def is_gemini(deployment: D) -> bool:
    return "gemini" in deployment.value


@pytest.fixture
def deployment(deployment_entry: deployment_entry) -> D:
    return deployment_entry.deployment


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture
def openai_client(deployment_entry: deployment_entry, get_openai_client):
    return get_openai_client(
        deployment_entry.upstream, headers=deployment_entry.source.headers()
    )


@pytest.fixture(
    params=[
        user_with_attachment_data,
        user_with_attachment_url,
        user_with_image_url,
    ],
    ids=[
        "attachment_data",
        "attachment_data_url",
        "content_part_image_url",
    ],
)
def create_message_with_image(request) -> Callable:
    return request.param


class Chat(Protocol):
    def __call__(
        self, **kwargs: Unpack[ChatCompletionArgs]
    ) -> Awaitable[ChatCompletionResult]: ...


@pytest.fixture
def chat(openai_client: openai.AsyncAzureOpenAI, stream: bool):
    async def _inner(
        **kwargs: Unpack[ChatCompletionArgs],
    ) -> ChatCompletionResult:
        return await chat_completion(openai_client, stream=stream, **kwargs)

    return _inner


def display_deployment(dep: deployment_entry):
    return dep.display()


@pytest.mark.parametrize(
    "deployment_entry", retired_deployments, ids=display_deployment
)
async def test_retired_models(deployment: D, chat: Chat):
    async with expected_exception(
        cls=openai.NotFoundError,
        status_code=404,
        message="not found",
    ):
        if is_vision_model(deployment):
            user_message = user_with_image_url(
                "describe the image", DOG_PICTURE
            )
        else:
            user_message = user("test")

        await chat(messages=[user_message], max_tokens=1)


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_model_field(deployment_entry: deployment_entry, chat: Chat):
    response = await chat(messages=[user("2+3=?")], max_tokens=1)
    assert deployment_entry.upstream == response.response.model


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_2_plus_3(chat: Chat):
    response = await chat(messages=[user("2+3=?")])
    assert "5" in response.content


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_hello(chat: Chat):
    response = await chat(messages=[user('Reply with "Hello"')])
    assert "hello" in response.content.lower()


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_empty_sys_message(chat: Chat):
    response = await chat(messages=[sys(""), user("2+4=?")])
    assert "6" in response.content.lower()


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_non_empty_sys_message(chat: Chat):
    system = sys("Act as helpful assistant")
    response = await chat(messages=[system, user("2+5=?")])
    assert "7" in response.content.lower()


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_empty_assistant_message(deployment: D, chat: Chat):
    messages = [
        user("hi, what is your name?"),
        ai(""),
        user("please come again?"),
    ]

    expected = None
    if is_claude(deployment):
        expected = ExpectedException(
            type=openai.BadRequestError,
            message="messages: text content blocks must contain non-whitespace text",
            status_code=400,
        )

    await _run_test(deployment, chat, messages, expected)


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_empty_user_message(deployment: D, chat: Chat):
    messages = [
        user(""),
        ai("come again?"),
        user("2+3=?"),
    ]

    expected = "5"
    if is_claude(deployment):
        expected = ExpectedException(
            type=openai.BadRequestError,
            message="messages: text content blocks must contain non-whitespace text",
            status_code=400,
        )

    await _run_test(deployment, chat, messages, expected)


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_multiple_candidates(deployment: D, chat: Chat):
    max_tokens = 10 if not supports_thinking(deployment) else 250
    # Gemini 2.0 rate-limits always fail on such concurrency
    n = 5 if not is_gemini(deployment) else 2

    response = await chat(
        messages=[user("2+7=? Reply with a single number")],
        max_tokens=max_tokens,
        n=n,
    )

    assert len(response.contents) == n
    for content in response.contents:
        assert "9" in content


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_finish_reason_length(deployment: D, chat: Chat):
    if is_gemini_image(deployment):
        pytest.skip(
            "Gemini Image doesn't seem to support max_tokens parameter."
        )

    response = await chat(
        max_tokens=1,
        messages=[
            user(
                "Tell me the full story of Pinocchio. Generate text and only the text."
            )
        ],
    )

    expected_tokens = 0 if supports_thinking(deployment) else 1
    assert len(response.content.split()) <= expected_tokens
    assert response.usage is not None
    assert response.usage.completion_tokens == expected_tokens
    assert response.finish_reasons == ["length"]


def _check_thinking_response(response: ChatCompletionResult):
    assert response.usage is not None, "Usage is missing"
    assert response.usage.completion_tokens > 10

    stages = response.stages
    assert stages is not None, "Stages are missing"
    assert len(stages) == 1

    thinking_stage = stages[0]
    assert thinking_stage.name == "Thinking"
    assert thinking_stage.content is not None, "Thinking content is missing"
    assert len(thinking_stage.content) > 10

    assert response.finish_reasons == ["stop"]


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_thinking), deployments),
    ids=display_deployment,
)
async def test_thinking_budget(chat: Chat):
    response = await chat(
        messages=[user("2+3=?")],
        configuration={
            "thinking": {"include_thoughts": True, "thinking_budget": 2048}
        },
    )

    assert "5" in response.content
    _check_thinking_response(response)


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_thinking_level), deployments),
    ids=display_deployment,
)
async def test_thinking_level(chat: Chat):
    thinking = {"include_thoughts": True, "thinking_level": "low"}
    response = await chat(
        messages=[user("2+3=?")], configuration={"thinking": thinking}
    )
    assert "5" in response.content
    _check_thinking_response(response)


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_thinking_level), deployments),
    ids=display_deployment,
)
async def test_reasoning_effort(chat: Chat):
    response = await chat(
        messages=[user("2+3=?")],
        configuration={"thinking": {"include_thoughts": True}},
        reasoning_effort="low",
    )
    assert "5" in response.content
    _check_thinking_response(response)


@pytest.mark.parametrize(
    "deployment_entry", deployments, ids=display_deployment
)
async def test_stop_sequence(deployment: D, stream: bool, chat: Chat):
    if is_gemini_image(deployment):
        pytest.skip("Gemini Image doesn't seem to support stop parameter.")

    if deployment == D.GEMINI_3_PRO_PREVIEW and not stream:
        pytest.skip(
            "Gemini 3 doesn't seem to support stop parameter in non-streaming mode."
        )

    stop = ["world"]
    response = await chat(
        max_tokens=None, stop=stop, messages=[user('Reply with "hello world"')]
    )
    content = response.content.lower()
    assert not all(w in content for w in stop)


@pytest.mark.parametrize(
    "deployment_entry", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("describe the image", DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment_entry", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_empty_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("", DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment_entry", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_without_text_part(deployment: D, chat: Chat):
    messages = [user_with_image_url(None, DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment_entry", vision_deployments, ids=display_deployment
)
async def test_vision_two_turns(
    deployment: D, chat: Chat, create_message_with_image
):
    if is_gemini_image(deployment):
        pytest.skip(
            "Gemini Image generates a variation of the given image with 2+3=5 text embedded into it, instead of describing the given image."
        )

    user_message = create_message_with_image("", DOG_PICTURE)
    messages = [
        sys("describe an image when you receive it"),
        user("2+3=?"),
        ai("5"),
        user_message,
    ]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment_entry", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_system(
    deployment: D, chat: Chat, create_message_with_image
):
    user_message = create_message_with_image(None, DOG_PICTURE)
    messages = [sys("describe the image"), user_message]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


async def _run_test(
    deployment: D,
    chat: Chat,
    messages,
    expected: str | List[str] | ExpectedException | None,
):
    async def _run():
        max_tokens = 2000 if supports_thinking(deployment) else 100
        return await chat(max_tokens=max_tokens, messages=messages)

    if isinstance(expected, ExpectedException):
        async with expected_exception(expected):
            await _run()
    else:
        response = await _run()
        response_content = response.content.lower()
        for stage in response.stages or []:
            if stage.content:
                response_content += "\n" + stage.content

        if expected is not None:
            substrings = [expected] if isinstance(expected, str) else expected
            assert any(s in response_content for s in substrings)


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_function_call(test: ToolCallTest, chat: Chat):
    response = await chat(
        messages=test.messages(True),
        functions=test.functions,
    )

    function_call = response.function_call
    assert function_call is not None, "Function call is missing"
    assert function_call.name == test.function_name

    function_args = json.loads(function_call.arguments)
    assert match_objects(test.expected_function_args(0), function_args)


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize("test", [ToolCallTest(1)], ids=lambda x: x.get_id())
async def test_function_response(test: ToolCallTest, chat: Chat):
    messages = [
        *test.messages(True),
        test.function_request(0),
        test.function_response(0),
    ]

    response = await chat(messages=messages, functions=test.functions)

    assert str(test.city_temps[0]) in response.content


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_undeclared_tool(deployment: D, chat: Chat):
    async def _run():
        return await chat(
            messages=[
                sys(
                    "Be a helpful assistant. You can call a tool called 'get_current_time' "
                    "when the user asks what time is it now."
                ),
                user("what time is it now?"),
            ]
        )

    message = (
        "The function call generated by the model is invalid:.*get_current_time"
    )

    if deployment in (
        D.GEMINI_2_5_PRO,
        D.GEMINI_2_5_FLASH,
        D.GEMINI_3_PRO_PREVIEW,
    ):
        async with expected_exception(
            [
                ExpectedException(
                    type=openai.InternalServerError,
                    status_code=500,
                    message=message,
                ),
                ExpectedException(type=openai.APIError, message=message),
            ]
        ):
            await _run()
    else:
        response = await _run()
        assert response.tool_calls is None
        assert response.finish_reasons == ["stop"]


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_tool_call_basic(deployment: D, test: ToolCallTest, chat: Chat):
    response = await chat(messages=test.messages(True), tools=test.tools)

    tool_calls = response.tool_calls
    assert tool_calls is not None, "Tool calls are missing"

    expected_calls = (
        test.targets if supports_parallel_tool_calls(deployment) else 1
    )

    assert (
        len(tool_calls) >= expected_calls
    ), f"Number of tools calls: actual ({len(tool_calls)}), expected ({expected_calls})"

    for idx, tool_call in enumerate(tool_calls):
        if not supports_tool_call_ids(deployment):
            name = f"{test.function_name}_{idx+1}"
            assert tool_call.id == name

        function_call = tool_call.function
        assert function_call.name == test.function_name

    # Arguments are needed to be sorted, because sometimes Claude produces tool calls out-of-order
    tool_args = sorted(
        [call.function.arguments for call in tool_calls],
        key=lambda x: json.dumps(json.loads(x), sort_keys=True),
    )

    for idx, args in enumerate(tool_args):
        function_args = json.loads(args)
        assert match_objects(test.expected_function_args(idx), function_args)


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_tool_response(test: ToolCallTest, chat: Chat):
    messages = [
        *test.messages(True),
        test.tool_request(),
        *test.tool_responses(),
    ]

    response = await chat(messages=messages, tools=test.tools)

    for temp in test.city_temps:
        assert str(temp) in response.content


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_and_response(deployment: D, chat: Chat):
    messages: List[ChatCompletionMessageParam] = [
        user("Tell me what's the temperature in London, UK in celsius?"),
    ]

    response = await chat(
        messages=messages, tools=[function_to_tool(GET_WEATHER_FUNCTION)]
    )

    assert response.finish_reasons == ["tool_calls"]
    assert response.tool_calls is not None, "Tool calls are missing"
    assert response.tool_calls[0].function.name == "get_temperature"
    tool_call_id = response.tool_calls[0].id

    response_message = response.response.choices[0].message.to_dict()

    if deployment == D.GEMINI_3_PRO_PREVIEW:
        dial_message = DialMessage.parse_obj(response_message)
        state = _parse_message_content_from_state(0, dial_message)
        assert state is not None, "state is missing"
        content = state.gemini_message_content
        assert content is not None, "gemini_message_content is missing"
        sig = (content.parts or [])[0].thought_signature
        assert sig is not None, "thought_signature is missing"

    messages.append(response_message)  # type: ignore
    messages.append(tool_response(tool_call_id, "it's 20 degrees celsius"))

    response = await chat(
        messages=messages, tools=[function_to_tool(GET_WEATHER_FUNCTION)]
    )

    assert "20" in response.content
    assert response.finish_reasons == ["stop"]


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_with_schema_references(chat: Chat):
    response = await chat(
        messages=[user("Tell me what's the temperature in London in celsius?")],
        tools=[GET_WEATHER_TOOL_WITH_REFERENCES],
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None, "Tool calls are missing"
    assert tool_calls[0].function.name == "get_temperature"


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_zero_parameters(chat: Chat):
    response = await chat(
        messages=[user("What time is it?")],
        tools=[function_to_tool(GET_CURRENT_TIME_FUNCTION)],
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None, "Tool calls are missing"
    assert tool_calls[0].type == "function"
    function = tool_calls[0].function
    assert function.name == "get_current_time"
    assert function.arguments == "{}"
    assert response.finish_reasons == ["tool_calls"]


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_calls_without_tool_definitions(deployment: D, chat: Chat):
    if deployment == D.CLAUDE_3_OPUS:
        pytest.skip(
            "Claude 3 Opus doesn't handle well inconsistent requests. "
            "It finishes with the stop reason `stop_sequence` where stop "
            "sequence is `<antml:function_calls>` which is a keyword from the Anthropic Claude system prompt."
        )

    response = await chat(
        messages=[
            user("what time is it?"),
            ai_tools(
                [
                    {
                        "type": "function",
                        "id": "tool-call-id1",
                        "function": {
                            "name": "get_current_time",
                            "arguments": "{}",
                        },
                    }
                ]
            ),
            tool_response(id="tool-call-id1", content="01:22 AM"),
            ai("It's 01:22 AM"),
            user("Now compute (2+3). Reply with a single digit"),
        ],
    )

    assert "5" in response.content
    assert response.finish_reasons == ["stop"]


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize("stream", [True], ids=["stream"])
@pytest.mark.parametrize(
    "description", ["", " \n\t", None], ids=["empty", "whitespace", "missing"]
)
async def test_tool_call_with_vacuous_description(
    description: str | None, chat: Chat
):
    func_def = GET_CURRENT_TIME_FUNCTION.copy()
    if description is None:
        func_def.pop("description")
    else:
        func_def["description"] = description

    response = await chat(
        messages=[user("what time is it?")],
        tools=[function_to_tool(func_def)],
    )
    assert response.finish_reasons == ["tool_calls"]
    assert response.tool_calls is not None, "Tool calls are missing"
    assert response.tool_calls[0].function.name == "get_current_time"


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_required(chat: Chat):
    response = await chat(
        messages=[user("How are you?")],
        tools=[
            function_to_tool(
                {
                    "name": "get_current_time",
                    "description": "return the current time",
                }
            )
        ],
        tool_choice="required",
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None, "Tool call is missing"
    assert tool_calls[0].function.name == "get_current_time"


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_choice_none(chat: Chat):
    response = await chat(
        messages=[user("What time is it?")],
        tools=[
            function_to_tool(
                {
                    "name": "get_current_time",
                    "description": "return the current time",
                }
            )
        ],
        tool_choice="none",
    )

    assert (
        response.tool_calls is None
    ), "No tools are expected to be called with tool_choice='none'"


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_json_object_response_format), deployments),
    ids=display_deployment,
)
async def test_json_object_response_format(chat: Chat):
    response = await chat(
        messages=[user("extract name and surname from 'John Doe'")],
        extra_body={"response_format": {"type": "json_object"}},
    )

    assert isinstance(json.loads(response.content), (dict, list))


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_json_schema_response_format), deployments),
    ids=display_deployment,
)
async def test_json_schema_response_format(chat: Chat):
    response = await chat(
        messages=[user("extract name and surname from 'John Doe'")],
        extra_body={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Schema name",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "NameField": {"type": "string"},
                            "SurnameField": {"type": "string"},
                        },
                    },
                },
            }
        },
    )

    assert json.loads(response.content) == {
        "NameField": "John",
        "SurnameField": "Doe",
    }


def _check_response_with_grounding(
    deployment: D, response: ChatCompletionResult, expected_content: str
):
    assert response.attachments is not None, "Attachments are missing"
    assert len(response.attachments) > 0
    assert isinstance(response.attachments[0].reference_url, str)
    assert response.attachments[0].reference_url.startswith(
        "https://vertexaisearch"
    )
    assert expected_content.lower() in response.content.lower()
    assert response.usage is not None, "Usage is missing"
    assert (
        response.usage.total_tokens > 7000
        if not is_gemini(deployment)
        else True
    )


@pytest.mark.parametrize(
    "deployment_entry",
    select(pred(supports_grounding), deployments),
    ids=display_deployment,
)
async def test_static_google_search(deployment: D, chat: Chat):
    response = await chat(
        messages=[user("Who won the Wimbledon in 2024?")],
        static_tools=StaticToolsConfig(
            functions=[
                StaticFunction(
                    name="google_search",
                    description="Search the web",
                    configuration={},
                ),
            ]
        ),
    )

    _check_response_with_grounding(deployment, response, "carlos alcaraz")


@pytest.mark.parametrize(
    "deployment_entry", sample_deployment, ids=display_deployment
)
async def test_allow_stream_options(chat: Chat):
    response = await chat(
        messages=[{"role": "user", "content": "2+2=?"}],
        max_tokens=10,
        extra_body={"stream_options": {"include_usage": True}},
    )
    assert "4" in response.content


@pytest.mark.parametrize(
    "deployment_entry", sample_deployment, ids=display_deployment
)
async def test_reject_extra_top_level_fields(chat: Chat):
    async with expected_exception(
        cls=openai.BadRequestError,
        status_code=400,
        message="Your request contained invalid structure on path extra-top-field. extra fields not permitted",
    ):
        await chat(
            messages=[{"role": "user", "content": "2+2=?"}],
            max_tokens=1,
            extra_body={"extra-top-field": "extra-top-value"},
        )


@pytest.mark.parametrize(
    "deployment_entry", sample_deployment, ids=display_deployment
)
async def test_reject_extra_message_fields(chat: Chat):
    async with expected_exception(
        cls=openai.BadRequestError,
        status_code=400,
        message="Your request contained invalid structure on path messages.0.extra-message-field. extra fields not permitted",
    ):
        extra_message = {"extra-message-field": "extra-message-value"}
        messages = [{"role": "user", "content": "2+2=?", **extra_message}]
        await chat(messages=messages, max_tokens=1)  # type: ignore


async def run_block_and_large_max_tokens_success(chat: Chat):
    """
    Testing behavior of Anthropic SDK in non-streaming mode with
    sufficiently large max_tokens value.
    """
    messages = [{"role": "user", "content": "2+3=?"}]
    return await chat(messages=messages, max_tokens=30_000)  # type: ignore


@pytest.mark.parametrize(
    "deployment_entry", sample_deployment, ids=display_deployment
)
@pytest.mark.parametrize("stream", [False], ids=["block"])
async def test_block_and_large_max_tokens_success(chat: Chat):
    response = await run_block_and_large_max_tokens_success(chat)
    assert "5" in response.content


@pytest.mark.parametrize(
    "deployment_entry", sample_deployment, ids=display_deployment
)
@pytest.mark.parametrize("stream", [False], ids=["block"])
async def test_block_and_large_max_tokens_fail(chat: Chat):
    with patch(
        "aidial_adapter_vertexai.app_config._get_default_anthropic_timeout",
        return_value=anthropic._constants.DEFAULT_TIMEOUT,
    ):
        with pytest.raises(openai.InternalServerError) as exc:
            await run_block_and_large_max_tokens_success(chat)

        e = exc.value
        assert e.status_code == 500
        assert e.body == {
            "code": "500",
            "type": "internal_server_error",
            "message": "Streaming is strongly recommended for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details",
        }
