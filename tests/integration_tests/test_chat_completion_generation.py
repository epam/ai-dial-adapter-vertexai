import json
from typing import Awaitable, Callable, List, Mapping, Protocol, Unpack

import openai
import pytest
from aidial_sdk.chat_completion.request import StaticFunction

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.integration_tests.constants import DOG_PICTURE, DOG_PICTURE_CONTENT
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.json import match_objects
from tests.utils.openai import (
    GET_WEATHER_TOOL_WITH_REFERENCES,
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    chat_completion,
    function_to_tool,
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
_GLOBAL = "global"

_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.GEMINI_2_0_FLASH_EXP: _CENTRAL,
    D.GEMINI_2_0_FLASH_001: _CENTRAL,
    D.GEMINI_2_5_PRO: _CENTRAL,
    D.GEMINI_2_5_PRO_PREVIEW_03_25: _CENTRAL,
    D.GEMINI_2_0_FLASH_LITE_1: _CENTRAL,
    D.GEMINI_2_5_FLASH: _CENTRAL,
    D.GEMINI_2_5_FLASH_IMAGE_PREVIEW: _GLOBAL,
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
        D.GEMINI_2_5_PRO,
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


def select(p: Selector[D], xs: List[D]) -> List[D]:
    return [x for x in xs if p(x)]


all_deployments = list(_DEPLOYMENT_TO_REGION.keys())
deployments = select(~pred(is_retired_model), all_deployments)
retired_deployments = select(pred(is_retired_model), all_deployments)
vision_deployments = select(pred(is_vision_model), deployments)


def supports_json_object_response_format(
    deployment: D,
) -> bool:
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
    ]


def supports_tool_call_ids(deployment: D) -> bool:
    return is_claude(deployment)


def supports_grounding(deployment: D) -> bool:
    return (
        "gemini" in deployment.value
        and deployment != D.GEMINI_2_5_FLASH_IMAGE_PREVIEW
    )


def supports_thinking(deployment: D) -> bool:
    return deployment in [D.GEMINI_2_5_PRO, D.GEMINI_2_5_FLASH]


def is_gemini(deployment: D) -> bool:
    return "gemini" in deployment.value


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


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
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


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
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


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_finish_reason_length(deployment: D, chat: Chat):
    if deployment == D.GEMINI_2_5_FLASH_IMAGE_PREVIEW:
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


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_thinking), deployments),
    ids=display_deployment,
)
async def test_thinking(deployment: D, chat: Chat):
    response = await chat(
        messages=[user("2+3=?")],
        configuration={
            "thinking": {
                "include_thoughts": True,
                "thinking_budget": 2048,
            }
        },
    )

    assert "5" in response.content

    assert response.usage is not None
    assert response.usage.completion_tokens > 10

    stages = response.stages
    assert stages is not None and len(stages) == 1

    thinking_stage = stages[0]
    assert thinking_stage.name == "Thinking"
    assert thinking_stage.content is not None
    assert len(thinking_stage.content) > 10

    assert response.finish_reasons == ["stop"]


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_stop_sequence(deployment: D, chat: Chat):
    if deployment == D.GEMINI_2_5_FLASH_IMAGE_PREVIEW:
        pytest.skip("Gemini Image doesn't seem to support stop parameter.")

    stop = ["world"]
    response = await chat(
        max_tokens=None, stop=stop, messages=[user('Reply with "hello world"')]
    )
    content = response.content.lower()
    assert not all(w in content for w in stop)


@pytest.mark.parametrize(
    "deployment", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("describe the image", DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_empty_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("", DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_without_text_part(deployment: D, chat: Chat):
    messages = [user_with_image_url(None, DOG_PICTURE)]
    await _run_test(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments, ids=display_deployment
)
async def test_vision_two_turns(
    deployment: D, chat: Chat, create_message_with_image
):
    if deployment == D.GEMINI_2_5_FLASH_IMAGE_PREVIEW:
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
    "deployment", vision_deployments, ids=display_deployment
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
        if expected is not None:
            substrings = [expected] if isinstance(expected, str) else expected
            assert any(s in response.content.lower() for s in substrings)


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
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_call_zero_parameters(chat: Chat):
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
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None, "Tool calls are missing"
    assert tool_calls[0].function.name == "get_current_time"


@pytest.mark.parametrize(
    "deployment",
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
    "deployment",
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
        if not is_gemini(deployment)
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
    "deployment", [D.CLAUDE_3_7_SONNET], ids=display_deployment
)
async def test_allow_stream_options(chat: Chat):
    response = await chat(
        messages=[{"role": "user", "content": "2+2=?"}],
        max_tokens=10,
        extra_body={"stream_options": {"include_usage": True}},
    )
    assert "4" in response.content


@pytest.mark.parametrize(
    "deployment", [D.CLAUDE_3_7_SONNET], ids=display_deployment
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
    "deployment", [D.CLAUDE_3_7_SONNET], ids=display_deployment
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
