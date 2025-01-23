import asyncio
import re
from dataclasses import dataclass
from typing import Callable, List

import pytest
from aidial_sdk.chat_completion.request import StaticFunction
from openai import APIError, RateLimitError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic.v1 import BaseModel

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ChatCompletionResult,
    ai,
    ai_function,
    ai_tools,
    blue_pic,
    chat_completion,
    for_all_choices,
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
    static_tools: StaticToolsConfig | None

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
    ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
    ChatCompletionDeployment.GEMINI_2_0_EXPERIMENTAL_1206,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_1219,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_OPUS,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET,
    ChatCompletionDeployment.CLAUDE_3_HAIKU,
    # ChatCompletionDeployment.CLAUDE_3_SONNET, # Turn on, not available yet
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
        ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
        ChatCompletionDeployment.GEMINI_2_0_EXPERIMENTAL_1206,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
        ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_OPUS,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET,
        ChatCompletionDeployment.CLAUDE_3_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_SONNET,
    ]


def supports_parallel_tool_calls(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
        ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_OPUS,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET,
        ChatCompletionDeployment.CLAUDE_3_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_SONNET,
    ]


def supports_tool_call_ids(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
        ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_OPUS,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET,
        ChatCompletionDeployment.CLAUDE_3_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_SONNET,
    ]


def supports_static_tools(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_PRO_1,
        ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
        ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
        ChatCompletionDeployment.GEMINI_FLASH_1_5_V1,
        ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
        ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
        ChatCompletionDeployment.GEMINI_2_0_EXPERIMENTAL_1206,
    ]


def supports_text_input(deployment: ChatCompletionDeployment) -> bool:
    return deployment != ChatCompletionDeployment.GEMINI_PRO_VISION_1


def supports_empty_content(deployment: ChatCompletionDeployment) -> bool:
    return is_codechat(deployment) or deployment in [
        ChatCompletionDeployment.CHAT_BISON_1,
        ChatCompletionDeployment.CHAT_BISON_2,
        ChatCompletionDeployment.CHAT_BISON_2_32K,
    ]


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_PRO_VISION_1,
        ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
        ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
        # Upstream returns 'claude-3-5-haiku-20241022 does not support images.'
        # ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
        # This model hallucinates on a the test image
        # ChatCompletionDeployment.CLAUDE_3_OPUS,
        ChatCompletionDeployment.CLAUDE_3_5_SONNET,
        ChatCompletionDeployment.CLAUDE_3_HAIKU,
        ChatCompletionDeployment.CLAUDE_3_SONNET,
    ]


def support_thinking(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_1219,
    ]


def is_gemini_2(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
        ChatCompletionDeployment.GEMINI_2_0_EXPERIMENTAL_1206,
        ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_1219,
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
        static_tools: StaticToolsConfig | None = None,
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
                static_tools,
            )
        )

    if supports_text_input(deployment):
        test_case(
            name="2+3=5",
            messages=[user("2+3=?")],
            expected=for_all_choices(lambda s: "5" in s),
        )

        test_case(
            name="model field",
            messages=[user("test")],
            max_tokens=1,
            expected=lambda s: s.response.model == deployment.value,
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
            name="empty assistant content",
            messages=[
                user("hi, what is your name?"),
                ai(""),
                user("please come again?"),
            ],
            expected=(
                expected_success
                if supports_empty_content(deployment)
                else ExpectedException(
                    type=UnprocessableEntityError,
                    message="Assistant message content must be present",
                    status_code=422,
                )
            ),
        )

        test_case(
            name="empty user content",
            messages=[
                user(""),
            ],
            expected=(
                expected_success
                if supports_empty_content(deployment)
                else ExpectedException(
                    type=UnprocessableEntityError,
                    message="User message content must be present",
                    status_code=422,
                )
            ),
        )

        test_case(
            name="max tokens 1",
            max_tokens=1,
            messages=[user("tell me the full story of Pinocchio")],
            expected=for_all_choices(
                lambda s: (
                    len(s.split()) == 1
                    if not support_thinking(deployment)
                    else len(s.split()) == 0
                )
            ),
        )
        # Gemini 2.0 rate-limits always fail on such concurrency
        candidates_count = 5 if not is_gemini_2(deployment) else 2
        test_case(
            name="multiple candidates",
            max_tokens=10 if not support_thinking(deployment) else 250,
            n=candidates_count,
            messages=[user("2+7=? Reply with a single number")],
            expected=for_all_choices(lambda s: "9" in s, candidates_count),
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

        city_config = (
            [[("Glasgow", 15)], [("Glasgow", 15), ("London", 20)]]
            if supports_parallel_tool_calls(deployment)
            else [[("Glasgow", 15)]]
        )

        for cities in city_config:
            function = GET_WEATHER_FUNCTION
            tool = function_to_tool(function)
            fun_name = function["name"]

            city_names = [name for name, _ in cities]
            city_temps = [temp for _, temp in cities]

            query = f"What's the temperature in {' and in '.join(city_names)} in celsius?"

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
                def _check(id: str) -> bool:
                    return (
                        f"{fun_name}_{idx+1}" == id
                        if not supports_tool_call_ids(deployment)
                        else True
                    )

                return _check

            expected_city_names = (
                city_names
                if supports_parallel_tool_calls(deployment)
                else city_names[:1]
            )

            test_case(
                name=f"weather tool {test_name_suffix}",
                messages=init_messages,
                tools=[tool],
                expected=lambda s, n=expected_city_names: all(
                    is_valid_tool_call(
                        s.tool_calls,
                        idx,
                        check_tool_call_id(idx),
                        fun_name,
                        check_fun_args(n[idx]),
                    )
                    for idx in range(len(n))
                ),
            )

            tool_reqs = ai_tools(
                [
                    tool_request(
                        create_tool_call_id(idx),
                        fun_name,
                        create_fun_args(name),
                    )
                    for idx, (name, _) in enumerate(cities)
                ]
            )
            tool_resps = [
                tool_response(create_tool_call_id(idx), f"{temp} celsius")
                for idx, (_, temp) in enumerate(cities)
            ]

            test_case(
                name=f"weather tool followup {test_name_suffix}",
                messages=[*init_messages, tool_reqs, *tool_resps],
                tools=[tool],
                expected=lambda s, t=city_temps: s.content_contains_all(t),
            )

    if supports_static_tools(deployment):
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
            expected=lambda s: (
                s.attachments is not None
                and len(s.attachments) > 0
                and isinstance(s.attachments[0].reference_url, str)
                and s.attachments[0].reference_url.startswith(
                    "https://vertexaisearch"
                )
                and "carlos alcaraz" in s.content.lower()
                and s.usage is not None
                and (
                    s.usage.total_tokens > 7000
                    if not is_gemini_2(deployment)
                    else True
                )
            ),
        )

        if not is_gemini_2(deployment):
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
                expected=lambda s: (
                    not s.attachments
                    and "4" in s.content
                    and s.usage is not None
                    and s.usage.total_tokens < 20
                ),
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
                    expected=lambda s: (
                        s.attachments is not None
                        and len(s.attachments) > 0
                        and isinstance(s.attachments[0].reference_url, str)
                        and s.attachments[0].reference_url.startswith(
                            "https://vertexaisearch"
                        )
                        and "4" in s.content.lower()
                        and s.usage is not None
                        and s.usage.total_tokens > 7000
                    ),
                )
    if support_thinking(deployment):
        test_case(
            name="thinking",
            messages=[user("2+2=?")],
            expected=lambda s: s.stages is not None
            and len(s.stages) == 1
            and s.stages[0].name == "Thought Process"
            and "4" in s.content,
        )
    return test_cases


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
        retries = 7
        delay = 5

        async def _retry_wait(
            retries: int, delay: int, e: APIError | RateLimitError
        ):
            if attempt < retries - 1:
                delay *= 2
                await asyncio.sleep(delay)
            else:
                raise e

        for attempt in range(retries):
            try:
                return await chat_completion(
                    client,
                    test.messages,
                    test.streaming,
                    test.stop,
                    test.max_tokens,
                    test.n,
                    test.functions,
                    test.tools,
                    test.static_tools,
                )
            except RateLimitError as e:
                await _retry_wait(retries, delay, e)
            except APIError as e:
                if e.code == "429":
                    await _retry_wait(retries, delay, e)
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
        actual_output = await run_chat_completion()
        assert test.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
