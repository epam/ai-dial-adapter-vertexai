import re
from dataclasses import dataclass
from typing import Callable, List

import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import TEST_SERVER_URL
from tests.utils.llm import (
    ai,
    assert_dialog,
    create_chat_model,
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
    messages: List[BaseMessage]
    expected: Callable[[List[str]], bool] | Exception

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
            expected=Exception(EMPTY_HISTORY_ERROR),
        ),
        TestCase(
            name="single system message",
            deployment=deployment,
            messages=[sys("Act as a helpful assistant")],
            expected=Exception(ONLY_SYS_MESSAGE_ERROR),
        ),
        TestCase(
            name="two system messages",
            deployment=deployment,
            messages=[
                sys("Act as a helpful assistant"),
                sys("Act as a tax accountant"),
                user("2+2=?"),
            ],
            expected=Exception(EXTRA_SYS_MESSAGE_ERROR),
        ),
        TestCase(
            name="single empty user message",
            deployment=deployment,
            messages=[user("")],
            expected=lambda _: True,
        ),
        TestCase(
            name="last empty user message",
            deployment=deployment,
            messages=[user("2+2=?"), ai("4"), user("")],
            expected=lambda _: True,
        ),
        TestCase(
            name="last message is not human",
            deployment=deployment,
            messages=[ai("5"), user("2+2=?"), ai("4")],
            expected=Exception(LAST_IS_NOT_HUMAN_ERROR),
        ),
        TestCase(
            name="three user messages in a row",
            deployment=deployment,
            messages=[user("2+3=?"), user("2+4=?"), user("2+5=?")],
            expected=Exception(INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR),
        ),
        TestCase(
            name="two user messages in a row",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?")],
            expected=Exception(INCORRECT_DIALOG_STRUCTURE_LEN_ERROR),
        ),
        TestCase(
            name="ai then user",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?"), user("2+4=?")],
            expected=Exception(INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR),
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
    streaming = False
    model = create_chat_model(
        TEST_SERVER_URL, test.deployment, streaming, max_tokens=None
    )

    if isinstance(test.expected, Exception):
        with pytest.raises(Exception) as exc_info:
            await assert_dialog(
                model=model,
                messages=test.messages,
                output_predicate=lambda s: True,
                streaming=streaming,
                stop=None,
            )

        assert isinstance(exc_info.value, openai.error.OpenAIError)
        assert exc_info.value.http_status == 422
        assert re.search(str(test.expected), str(exc_info.value))
    else:
        await assert_dialog(
            model=model,
            messages=test.messages,
            output_predicate=test.expected,
            streaming=streaming,
            stop=None,
        )
