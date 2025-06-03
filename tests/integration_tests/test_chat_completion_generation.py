import asyncio
import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Mapping, Unpack

import openai
import pytest
from aidial_sdk.chat_completion.request import StaticFunction
from openai import APIError, RateLimitError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.integration_tests.constants import DOG_PICTURE
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    ai_function,
    ai_tools,
    assert_eq,
    assert_in,
    assert_not_in,
    chat_completion,
    for_all_choices,
    foreach,
    function_request,
    function_response,
    function_to_tool,
    is_valid_function_call,
    is_valid_tool_call,
    sanitize_test_name,
    sys,
    tool_request,
    tool_response,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)
from tests.utils.selector import Selector, pred


def expected_success(*args, **kwargs):
    pass


@dataclass
class TestCase:
    __test__ = False

    name: str
    region: str | None
    deployment: D
    streaming: bool

    messages: List[ChatCompletionMessageParam]
    expected: Callable[[ChatCompletionResult], None] | ExpectedException

    max_tokens: int | None
    stop: List[str] | None

    n: int | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None
    static_tools: StaticToolsConfig | None
    extra_body: dict | None

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value}/{self.streaming}/{self.name}"
        )


_CENTRAL = "us-central1"
_EAST = "us-east5"
_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.CHAT_BISON_1: _CENTRAL,
    D.CHAT_BISON_2_32K: _CENTRAL,
    D.CODECHAT_BISON_1: _CENTRAL,
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
        D.GEMINI_PRO_1,
        D.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
        D.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
        D.GEMINI_2_5_PRO_EXP_03_25,
        D.GEMINI_PRO_VISION_1,
        D.GEMINI_PRO_1_5_PREVIEW,
        D.GEMINI_PRO_1_5_V1,
        D.GEMINI_FLASH_1_5_V1,
        D.CHAT_BISON_1,
        D.CODECHAT_BISON_1,
        D.CHAT_BISON_2_32K,
    }


def select(p: Selector[D], xs: List[D]) -> List[D]:
    return [x for x in xs if p(x)]


all_deployments = list(_DEPLOYMENT_TO_REGION.keys())
deployments = select(~pred(is_retired_model), all_deployments)
retired_deployments = select(pred(is_retired_model), all_deployments)


def is_codechat(deployment: D) -> bool:
    return deployment in [
        D.CODECHAT_BISON_1,
        D.CODECHAT_BISON_2,
        D.CODECHAT_BISON_2_32K,
    ]


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


def supports_json_schema_response_format(
    deployment: D,
) -> bool:
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
    return "gemini" in deployment.value and deployment != D.GEMINI_PRO_VISION_1


def supports_empty_content(deployment: D) -> bool:
    return is_codechat(deployment) or deployment in [
        D.CHAT_BISON_1,
        D.CHAT_BISON_2,
        D.CHAT_BISON_2_32K,
    ]


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


def support_explicit_thinking(deployment: D) -> bool:
    return deployment in [
        D.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
    ]


def support_thinking(deployment: D) -> bool:
    return support_explicit_thinking(deployment) or deployment in [
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
async def test_empty_assistant_message(deployment: D, chat: Chat):
    messages = [
        user("hi, what is your name?"),
        ai(""),
        user("please come again?"),
    ]

    if not supports_empty_content(deployment):
        async with expected_exception(
            cls=UnprocessableEntityError,
            message="Assistant message content must be present",
            status_code=422,
        ):
            await chat(messages=messages)
    else:
        await chat(messages=messages)


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


def get_test_cases(
    deployment: D, region: str, streaming: bool
) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            Callable[[ChatCompletionResult], None] | ExpectedException
        ) = expected_success,
        n: int | None = None,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
        static_tools: StaticToolsConfig | None = None,
        extra_body: dict | None = None,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                region,
                deployment,
                streaming,
                messages,
                expected,
                max_tokens,
                stop,
                n,
                functions,
                tools,
                static_tools,
                extra_body,
            )
        )

    if is_retired_model(deployment):
        return []

    # Gemini 2.0 rate-limits always fail on such concurrency
    candidates_count = 5 if not is_gemini_2(deployment) else 2
    test_case(
        name="multiple candidates",
        max_tokens=10 if not support_thinking(deployment) else 250,
        n=candidates_count,
        messages=[user("2+7=? Reply with a single number")],
        expected=for_all_choices(lambda s: assert_in("9", s), candidates_count),
    )

    # Stop sequences do not work for some reason for CHAT_BISON_2_32K and streaming mode
    if (deployment, streaming) != (
        D.CHAT_BISON_2_32K,
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
                else for_all_choices(
                    lambda s: assert_not_in("world", s.lower())
                )
            ),
        )

    if is_vision_model(deployment):
        content = "describe the image"
        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, DOG_PICTURE),
                user_with_attachment_url(content, DOG_PICTURE),
                user_with_image_url(content, DOG_PICTURE),
            ]
        ):
            test_case(
                name=f"describe image {idx}",
                max_tokens=1000 if support_thinking(deployment) else 100,
                messages=[sys("be a helpful assistant"), user_message],
                expected=lambda s: assert_in("dog", s.content.lower()),
            )

    if supports_tools(deployment):

        city_config = (
            [
                [("Glasgow", "Scotland", 15)],
                [("Glasgow", "Scotland", 15), ("London", "England", 20)],
            ]
            if supports_parallel_tool_calls(deployment)
            else [[("Glasgow", "Scotland", 15)]]
        )

        for cities in city_config:
            function = GET_WEATHER_FUNCTION
            tool = function_to_tool(function)
            fun_name = function["name"]

            city_names = [name for name, _, _ in cities]
            city_countries = [country for _, country, _ in cities]
            city_temps = [temp for _, _, temp in cities]

            location_queries = [
                f"{name} in {country}"
                for name, country in zip(city_names, city_countries)
            ]

            query = f"What's the temperature in city of {' and in '.join(location_queries)} in celsius?"

            init_messages = [
                user("2+3=?"),
                ai("5"),
                user(query),
            ]

            init_messages.insert(0, sys("act as a helpful assistant"))

            def create_fun_args(city: str):
                return {
                    "location": city,
                    "format": "celsius",
                }

            def check_fun_args(city: str):
                return {
                    "location": lambda s: city.lower() in s.lower(),
                    "format": "celsius",
                }

            test_name_suffix = " ".join(city_names)

            # Functions
            test_case(
                name=f"weather function {test_name_suffix}",
                messages=init_messages,
                functions=[function],
                expected=lambda s, n=city_names[0]: is_valid_function_call(
                    s.function_call, fun_name, check_fun_args(n)
                ),
            )

            function_req = ai_function(
                function_request(fun_name, create_fun_args(city_names[0]))
            )
            function_resp = function_response(
                fun_name, f"{city_temps[0]} celsius"
            )

            if len(cities) == 1:
                test_case(
                    name=f"weather function followup {test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, t=city_temps[0]: s.content_contains_all(
                        [t]
                    ),
                )
            else:
                test_case(
                    name=f"weather function followup {test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, n=city_names[1]: is_valid_function_call(
                        s.function_call, fun_name, check_fun_args(n)
                    ),
                )

            # Tools
            def create_tool_call_id(idx: int):
                return f"{fun_name}_{idx+1}"

            def check_tool_call_id(idx: int):
                def ret(id: str):
                    if not supports_tool_call_ids(deployment):
                        assert f"{fun_name}_{idx+1}" == id

                return ret

            expected_city_names = (
                city_names
                if supports_parallel_tool_calls(deployment)
                else city_names[:1]
            )

            test_case(
                name=f"weather tool {test_name_suffix}",
                messages=init_messages,
                tools=[tool],
                expected=lambda s, n=expected_city_names: foreach(
                    lambda idx: is_valid_tool_call(
                        s.tool_calls,
                        idx,
                        check_tool_call_id(idx),
                        fun_name,
                        check_fun_args(n[idx]),
                    ),
                    range(len(n)),
                ),
            )

            tool_reqs = ai_tools(
                [
                    tool_request(
                        create_tool_call_id(idx),
                        fun_name,
                        create_fun_args(name),
                    )
                    for idx, (name, _, _) in enumerate(cities)
                ]
            )
            tool_resps = [
                tool_response(create_tool_call_id(idx), f"{temp} celsius")
                for idx, (_, _, temp) in enumerate(cities)
            ]

            test_case(
                name=f"weather tool followup {test_name_suffix}",
                messages=[*init_messages, tool_reqs, *tool_resps],
                tools=[tool],
                expected=lambda s, t=city_temps: s.content_contains_all(t),
            )

    if supports_grounding(deployment):

        def _checker(s: ChatCompletionResult):
            assert s.attachments is not None
            assert len(s.attachments) > 0
            assert isinstance(s.attachments[0].reference_url, str)
            assert s.attachments[0].reference_url.startswith(
                "https://vertexaisearch"
            )
            assert "carlos alcaraz" in s.content.lower()
            assert s.usage is not None
            assert (
                s.usage.total_tokens > 7000
                if not is_gemini_2(deployment)
                else True
            )

        test_case(
            name="static google search",
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
            expected=_checker,
        )

        if not is_gemini_2(deployment):

            def _simple_checker(s: ChatCompletionResult):
                assert not s.attachments
                assert "4" in s.content
                assert s.usage is not None
                assert s.usage.total_tokens < 20

            test_case(
                name="static google search with dynamic threshold not hit",
                messages=[user("2+2=?")],
                static_tools=StaticToolsConfig(
                    functions=[
                        StaticFunction(
                            name="google_search",
                            description="Search the web",
                            configuration={
                                "dynamic_retrieval_config": {
                                    "mode": "MODE_DYNAMIC",
                                    "dynamic_threshold": 0.8,
                                }
                            },
                        ),
                    ]
                ),
                max_tokens=100,
                expected=_simple_checker,
            )

            for index, retrieval_config in enumerate(
                [
                    {"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.01},
                    {"mode": "MODE_UNSPECIFIED"},
                ]
            ):
                test_case(
                    name=f"static google search with guaranteed search {index}",
                    messages=[user("2+2=")],
                    static_tools=StaticToolsConfig(
                        functions=[
                            StaticFunction(
                                name="google_search",
                                description="Search the web",
                                configuration={
                                    "dynamic_retrieval_config": retrieval_config
                                },
                            ),
                        ]
                    ),
                    max_tokens=100,
                    expected=_checker,
                )

    if support_explicit_thinking(deployment):

        def _checker(s: ChatCompletionResult):
            assert s.stages is not None
            assert len(s.stages) == 1
            assert s.stages[0].name == "Thought Process"
            assert "4" in s.content

        test_case(
            name="thinking",
            messages=[user("2+2=?")],
            expected=_checker,
        )

    if supports_json_object_response_format(deployment):

        def _checker(s: ChatCompletionResult):
            assert isinstance(json.loads(s.content), (dict, list))

        test_case(
            name="response format json object",
            messages=[user("extract name and surname from 'John Doe'")],
            extra_body={"response_format": {"type": "json_object"}},
            expected=_checker,
        )

    if supports_json_schema_response_format(deployment):
        test_case(
            name="response format json schema",
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
            expected=lambda s: assert_eq(
                json.loads(s.content),
                {
                    "NameField": "John",
                    "SurnameField": "Doe",
                },
            ),
        )

    return test_cases


@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment, region in _DEPLOYMENT_TO_REGION.items()
        for streaming in [False, True]
        for test in get_test_cases(deployment, region, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion(get_openai_client, test: TestCase):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        test.deployment.value, region=test.region
    )

    async def run_chat_completion() -> ChatCompletionResult:
        max_attempts = 6
        delay = 1

        async def _retry_wait(
            is_last_attempt: bool, e: APIError | RateLimitError
        ):
            if is_last_attempt:
                raise e

            nonlocal delay
            await asyncio.sleep(delay)
            delay *= 2

        for attempt in range(max_attempts):
            is_last_attempt = attempt == max_attempts - 1
            try:
                return await chat_completion(
                    client,
                    messages=test.messages,
                    stream=test.streaming,
                    stop=test.stop,
                    max_tokens=test.max_tokens,
                    n=test.n,
                    functions=test.functions,
                    tools=test.tools,
                    static_tools=test.static_tools,
                    extra_body=test.extra_body,
                )
            except RateLimitError as e:
                await _retry_wait(is_last_attempt, e)
            except APIError as e:
                if e.code == "429":
                    await _retry_wait(is_last_attempt, e)
                else:
                    raise e
        raise RuntimeError("Failed to get a valid response")

    if isinstance(test.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        assert isinstance(
            actual_exc, test.expected.type
        ), f"Actual exception type ({type(actual_exc)}) doesn't match the expected one ({test.expected.type})"
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test.expected.status_code
        assert re.search(test.expected.message, str(actual_exc))
    else:
        try:
            actual_output = await run_chat_completion()
        except openai.APIError as e:
            assert False, str(e.body)
        else:
            assert test.expected(
                actual_output
            ), f"Failed output test, actual output: {actual_output}"
