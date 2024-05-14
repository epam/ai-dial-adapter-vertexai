import re
from dataclasses import dataclass
from typing import List

import pytest
from openai import BadRequestError, UnprocessableEntityError
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import TEST_SERVER_URL
from tests.utils.openai import (
    ChatCompletionResult,
    ai,
    chat_completion,
    get_client,
    sanitize_test_name,
    sys,
    user,
)

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment
    messages: List[ChatCompletionMessageParam]
    expected_exception: Exception | None

    def get_id(self) -> str:
        return sanitize_test_name(f"{self.deployment.value} {self.name}")


EMPTY_HISTORY_ERROR = "The chat history must have at least one message"
ONLY_SYS_MESSAGE_ERROR = (
    "The chat history must have at least one non-system message"
)
EXTRA_SYS_MESSAGE_ERROR = (
    "System messages other than the initial system message are not allowed"
)
LAST_IS_NOT_HUMAN_ERROR = "The last message must be a user message"
INCORRECT_DIALOG_STRUCTURE_LEN_ERROR = (
    "There should be odd number of messages for correct alternating turn"
)
INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR = (
    "Messages must alternate between authors"
)


def get_test_cases(
    deployment: ChatCompletionDeployment,
) -> List[TestCase]:
    return [
        TestCase(
            name="empty history",
            deployment=deployment,
            messages=[],
            expected_exception=Exception(EMPTY_HISTORY_ERROR),
        ),
        TestCase(
            name="single system message",
            deployment=deployment,
            messages=[sys("Act as a helpful assistant")],
            expected_exception=Exception(ONLY_SYS_MESSAGE_ERROR),
        ),
        TestCase(
            name="two system messages",
            deployment=deployment,
            messages=[
                sys("Act as a helpful assistant"),
                sys("Act as a tax accountant"),
                user("2+2=?"),
            ],
            expected_exception=Exception(EXTRA_SYS_MESSAGE_ERROR),
        ),
        TestCase(
            name="single empty user message",
            deployment=deployment,
            messages=[user("")],
            expected_exception=None,
        ),
        TestCase(
            name="last empty user message",
            deployment=deployment,
            messages=[user("2+2=?"), ai("4"), user("")],
            expected_exception=None,
        ),
        TestCase(
            name="last message is not human",
            deployment=deployment,
            messages=[ai("5"), user("2+2=?"), ai("4")],
            expected_exception=Exception(LAST_IS_NOT_HUMAN_ERROR),
        ),
        TestCase(
            name="three user messages in a row",
            deployment=deployment,
            messages=[user("2+3=?"), user("2+4=?"), user("2+5=?")],
            expected_exception=Exception(
                INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR
            ),
        ),
        TestCase(
            name="two user messages in a row",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?")],
            expected_exception=Exception(INCORRECT_DIALOG_STRUCTURE_LEN_ERROR),
        ),
        TestCase(
            name="ai then user",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?"), user("2+4=?")],
            expected_exception=Exception(
                INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR
            ),
        ),
    ]


validation_test_cases: List[TestCase] = [
    test_case
    for deployment in deployments
    for test_case in get_test_cases(deployment)
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test", validation_test_cases, ids=lambda test: test.get_id()
)
async def test_input_validation(server, test: TestCase):
    client = get_client(TEST_SERVER_URL, test.deployment.value)

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(
            client, test.messages, False, None, None, None, None, None
        )

    if test.expected_exception is not None:
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        assert isinstance(exc_info.value, UnprocessableEntityError)
        assert re.search(str(test.expected_exception), str(exc_info.value))
    else:
        await run_chat_completion()


@pytest.mark.asyncio
async def test_imagen_content_filtering(server):
    client = get_client(
        TEST_SERVER_URL, ChatCompletionDeployment.IMAGEN_005.value
    )
    messages: List[ChatCompletionMessageParam] = [
        user("generate something unsafe")
    ]

    with pytest.raises(Exception) as exc_info:
        await chat_completion(
            client, messages, False, None, None, None, None, None
        )

    assert isinstance(exc_info.value, BadRequestError)

    resp = exc_info.value.response.json()
    assert (resp["error"]["code"]) == "content_filter"
    assert (
        resp["error"]["message"]
        == "The response is blocked, as it may violate our policies."
    )
