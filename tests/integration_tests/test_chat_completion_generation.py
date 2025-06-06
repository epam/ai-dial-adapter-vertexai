import json
from typing import Awaitable, Callable, List, Mapping, Unpack

import openai
import pytest
from aidial_sdk.chat_completion.request import StaticFunction
from openai import UnprocessableEntityError

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.integration_tests.constants import DOG_PICTURE
from tests.utils.exception import expected_exception
from tests.utils.json import match_objects
from tests.utils.openai import (
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    chat_completion,
    sanitize_test_name,
    sys,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)
from tests.utils.selector import Selector, pred
from tests.utils.tools import ToolCallTest

_CENTRAL = "us-central1"
_EAST = "us-east5"
_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.CHAT_BISON_1: _CENTRAL,
    D.CHAT_BISON_2: _CENTRAL,
    D.CHAT_BISON_2_32K: _CENTRAL,
    D.CODECHAT_BISON_1: _CENTRAL,
    D.CODECHAT_BISON_2: _CENTRAL,
    D.CODECHAT_BISON_2_32K: _CENTRAL,
    D.GEMINI_PRO_1: _CENTRAL,
    D.GEMINI_FLASH_1_5_V2: _CENTRAL,
    D.GEMINI_PRO_VISION_1: _CENTRAL,
    D.GEMINI_PRO_1_5_V2: _CENTRAL,
    D.GEMINI_2_0_FLASH_EXP: _CENTRAL,
    D.GEMINI_2_0_FLASH_001: _CENTRAL,
    D.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: _CENTRAL,
    D.GEMINI_2_5_PRO_EXP_03_25: _CENTRAL,
    D.GEMINI_2_0_FLASH_THINKING_EXP_01_21: _CENTRAL,
    D.GEMINI_2_0_FLASH_LITE_1: _CENTRAL,
    D.GEMINI_2_5_FLASH_PREVIEW_04_17: _CENTRAL,
    D.CLAUDE_3_5_SONNET_V2: _EAST,
    D.CLAUDE_3_5_HAIKU: _EAST,
    D.CLAUDE_3_OPUS: _EAST,
    D.CLAUDE_3_5_SONNET: _EAST,
    D.CLAUDE_3_HAIKU: _EAST,
    D.CLAUDE_3_7_SONNET: _EAST,
    D.CLAUDE_4_SONNET: _EAST,
    D.CLAUDE_4_OPUS: _EAST,
}


def is_retired_model(deployment: D) -> bool:
    # Keep at least one model in the list to test how the adapter handles retired models in streaming and non-streaming modes
    return deployment in {
        D.CHAT_BISON_1,
        D.CHAT_BISON_2,
        D.CHAT_BISON_2_32K,
        D.CODECHAT_BISON_1,
        D.CODECHAT_BISON_2,
        D.CODECHAT_BISON_2_32K,
        D.GEMINI_PRO_1,
        D.GEMINI_PRO_VISION_1,
        D.GEMINI_PRO_1_5_PREVIEW,
        D.GEMINI_PRO_1_5_V1,
        D.GEMINI_FLASH_1_5_V1,
        D.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
        D.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
        D.GEMINI_2_5_PRO_EXP_03_25,
    }


def select(p: Selector[D], xs: List[D]) -> List[D]:
    return [x for x in xs if p(x)]


all_deployments = list(_DEPLOYMENT_TO_REGION.keys())
deployments = select(~pred(is_retired_model), all_deployments)
retired_deployments = select(pred(is_retired_model), all_deployments)


def supports_json_object_response_format(
    deployment: D,
) -> bool:
    return deployment in [
        D.GEMINI_PRO_1,
        D.GEMINI_PRO_1_5_PREVIEW,
        D.GEMINI_PRO_1_5_V1,
        D.GEMINI_PRO_1_5_V2,
        D.GEMINI_FLASH_1_5_V1,
        D.GEMINI_FLASH_1_5_V2,
        D.GEMINI_2_0_FLASH_EXP,
        D.GEMINI_2_0_FLASH_001,
    ]


def supports_json_schema_response_format(deployment: D) -> bool:
    return supports_json_object_response_format(
        deployment
    ) and deployment not in [
        D.GEMINI_PRO_1,
    ]


def is_claude(deployment: D) -> bool:
    return "claude" in deployment.value


def supports_tools(deployment: D) -> bool:
    return is_claude(deployment) or deployment in [
        D.GEMINI_PRO_1,
        D.GEMINI_PRO_1_5_V1,
        D.GEMINI_2_0_FLASH_EXP,
        D.GEMINI_2_0_FLASH_001,
        D.GEMINI_2_0_PRO_EXP_02_05,
        D.GEMINI_2_5_PRO_EXP_03_25,
        D.GEMINI_2_0_FLASH_LITE_1,
        D.GEMINI_2_5_FLASH_PREVIEW_04_17,
    ]


def supports_parallel_tool_calls(deployment: D) -> bool:
    return deployment in [
        # D.CLAUDE_3_5_SONNET_V2,
        # D.CLAUDE_3_HAIKU,
        D.CLAUDE_3_5_HAIKU,
        D.CLAUDE_3_OPUS,
        D.CLAUDE_3_5_SONNET,
        # D.CLAUDE_3_7_SONNET,
        D.GEMINI_2_5_PRO_EXP_03_25,
        D.GEMINI_2_0_FLASH_LITE_1,
        D.GEMINI_2_5_FLASH_PREVIEW_04_17,
    ]


def supports_tool_call_ids(deployment: D) -> bool:
    return is_claude(deployment)


def supports_grounding(deployment: D) -> bool:
    return "gemini" in deployment.value


def is_vision_model(deployment: D) -> bool:
    return deployment in [
        D.GEMINI_PRO_VISION_1,
        D.GEMINI_PRO_1_5_V2,
        D.GEMINI_FLASH_1_5_V2,
        D.GEMINI_2_5_PRO_EXP_03_25,
        D.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
        D.GEMINI_2_0_PRO_EXP_02_05,
        D.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
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
    ]


def support_thinking(deployment: D) -> bool:
    return deployment in [
        # Gemini 2.5 doesn't emit thinking tokens into a separate output,
        # it's all the part of the completion tokens.
        D.GEMINI_2_5_PRO_EXP_03_25,
        D.GEMINI_2_5_FLASH_PREVIEW_04_17,
    ]


def is_gemini_2(deployment: D) -> bool:
    return "gemini-2." in deployment.value


@pytest.fixture
def deployment(request) -> D:
    return request.param


@pytest.fixture
def region(deployment: D) -> str:
    region = _DEPLOYMENT_TO_REGION.get(deployment)
    if region is None:
        raise ValueError(
            f"{deployment.value!r} is missing from the region mapping"
        )
    return region


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture
def openai_client(deployment: D, region: str, get_openai_client):
    return get_openai_client(deployment.value, region=region)


Chat = Callable[..., Awaitable[ChatCompletionResult]]


@pytest.fixture
def chat(openai_client: openai.AsyncAzureOpenAI, stream: bool):
    async def _inner(
        **kwargs: Unpack[ChatCompletionArgs],
    ) -> ChatCompletionResult:
        return await chat_completion(openai_client, stream=stream, **kwargs)

    return _inner


def display_deployment(dep: D):
    return sanitize_test_name(dep.value)


@pytest.mark.parametrize(
    "deployment", retired_deployments, ids=display_deployment
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


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_model_field(deployment: D, chat: Chat):
    response = await chat(messages=[user("test")], max_tokens=1)
    assert deployment.value == response.response.model


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_2_plus_3(chat: Chat):
    response = await chat(messages=[user("2+3=?")])
    assert "5" in response.content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_hello(chat: Chat):
    response = await chat(messages=[user('Reply with "Hello"')])
    assert "hello" in response.content.lower()


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_empty_sys_message(chat: Chat):
    response = await chat(messages=[sys(""), user("2+4=?")])
    assert "6" in response.content.lower()


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_non_empty_sys_message(chat: Chat):
    system = sys("Act as helpful assistant")
    response = await chat(messages=[system, user("2+5=?")])
    assert "7" in response.content.lower()


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_empty_assistant_message(chat: Chat):
    messages = [
        user("hi, what is your name?"),
        ai(""),
        user("please come again?"),
    ]

    async with expected_exception(
        cls=UnprocessableEntityError,
        message="Assistant message content must be present",
        status_code=422,
    ):
        await chat(messages=messages)


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_multiple_candidates(deployment: D, chat: Chat):
    max_tokens = 10 if not support_thinking(deployment) else 250
    # Gemini 2.0 rate-limits always fail on such concurrency
    n = 5 if not is_gemini_2(deployment) else 2

    response = await chat(
        messages=[user("2+7=? Reply with a single number")],
        max_tokens=max_tokens,
        n=n,
    )

    assert len(response.contents) == n
    for content in response.contents:
        assert "9" in content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_finish_reason_length(deployment: D, chat: Chat):
    response = await chat(
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
    )

    expected_tokens = 0 if support_thinking(deployment) else 1
    assert len(response.content.split()) <= expected_tokens
    assert response.usage is not None
    assert response.usage.completion_tokens == expected_tokens
    assert response.finish_reasons == ["length"]


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_stop_sequence(chat: Chat):
    stop = ["world"]
    response = await chat(
        max_tokens=None, stop=stop, messages=[user('Reply with "hello world"')]
    )
    content = response.content.lower()
    assert not all(w in content for w in stop)


@pytest.mark.parametrize(
    "deployment",
    select(pred(is_vision_model), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "message_factory",
    [
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
async def test_vision(deployment: D, chat: Chat, message_factory):
    user_message = message_factory("describe the image", DOG_PICTURE)
    messages = [sys("be a helpful assistant"), user_message]
    max_tokens = 1000 if support_thinking(deployment) else 100

    response = await chat(max_tokens=max_tokens, messages=messages)
    assert "dog" in response.content.lower()


@pytest.mark.parametrize(
    "deployment",
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
    "deployment",
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
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_tool_call(deployment: D, test: ToolCallTest, chat: Chat):

    response = await chat(
        messages=test.messages(True),
        tools=test.tools,
    )

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

        function_args = json.loads(function_call.arguments)
        assert match_objects(test.expected_function_args(idx), function_args)


@pytest.mark.parametrize(
    "deployment",
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
    "deployment",
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
    "deployment",
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
        if not is_gemini_2(deployment)
        else True
    )


@pytest.mark.parametrize(
    "deployment",
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
    "deployment",
    select(pred(supports_grounding) & ~pred(is_gemini_2), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "search_config",
    [
        {"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.01},
        {"mode": "MODE_UNSPECIFIED"},
    ],
    ids=["dynamic-mode", "unspecified-mode"],
)
async def test_static_google_search_with_dynamic_config(
    deployment: D, chat: Chat, search_config: dict
):

    response = await chat(
        messages=[user("2+2=?")],
        static_tools=StaticToolsConfig(
            functions=[
                StaticFunction(
                    name="google_search",
                    description="Search the web",
                    configuration={"dynamic_retrieval_config": search_config},
                ),
            ]
        ),
        max_tokens=100,
    )

    _check_response_with_grounding(deployment, response, "4")
