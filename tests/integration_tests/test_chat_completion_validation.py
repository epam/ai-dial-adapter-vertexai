import re
from dataclasses import dataclass
from typing import List

import openai
import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from client.client_adapter import create_chat_model
from tests.conftest import TEST_SERVER_URL
from tests.utils import ai, assert_dialog, sanitize_test_name, sys, user

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
    expected_error: str

    def get_id(self) -> str:
        return sanitize_test_name(f"{self.deployment.value} {self.name}")


EMPTY_MESSAGE_ERROR = "Empty messages are not allowed"
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
            expected_error=EMPTY_HISTORY_ERROR,
        ),
        TestCase(
            name="single system message",
            deployment=deployment,
            messages=[sys("Act as a helpful assistant")],
            expected_error=ONLY_SYS_MESSAGE_ERROR,
        ),
        TestCase(
            name="two system messages",
            deployment=deployment,
            messages=[
                sys("Act as a helpful assistant"),
                sys("Act as a tax accountant"),
                user("2+2=?"),
            ],
            expected_error=EXTRA_SYS_MESSAGE_ERROR,
        ),
        TestCase(
            name="single empty user message",
            deployment=deployment,
            messages=[user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
        TestCase(
            name="last empty user message",
            deployment=deployment,
            messages=[user("2+2=?"), ai("4"), user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
        TestCase(
            name="last message is not human",
            deployment=deployment,
            messages=[ai("5"), user("2+2=?"), ai("4")],
            expected_error=LAST_IS_NOT_HUMAN_ERROR,
        ),
        TestCase(
            name="three user messages in a row",
            deployment=deployment,
            messages=[user("2+3=?"), user("2+4=?"), user("2+5=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR,
        ),
        TestCase(
            name="two user messages in a row",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_LEN_ERROR,
        ),
        TestCase(
            name="ai then user",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?"), user("2+4=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR,
        ),
    ]


validation_test_cases: List[TestCase] = [
    test_case
    for deployment in deployments
    for test_case in get_test_cases(deployment)
] + [
    TestCase(
        name="system message in codechat",
        deployment=ChatCompletionDeployment.CODECHAT_BISON_1,
        messages=[sys("Act as a helpful assistant"), user("2+2=?")],
        expected_error="System message is not supported",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test", validation_test_cases, ids=lambda test: test.get_id()
)
async def test_input_validation(server, test: TestCase):
    streaming = False
    model = create_chat_model(TEST_SERVER_URL, test.deployment, streaming)

    with pytest.raises(Exception) as exc_info:
        await assert_dialog(model, test.messages, lambda s: True, streaming)

    assert isinstance(exc_info.value, openai.error.OpenAIError)
    assert exc_info.value.http_status == 422
    assert re.search(test.expected_error, str(exc_info.value))
