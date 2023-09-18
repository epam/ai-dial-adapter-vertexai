import re
from dataclasses import dataclass
from typing import Any, List

import openai
import openai.error
import pytest
import requests
from langchain.schema import BaseMessage

from client.client_adapter import create_chat_model
from llm.vertex_ai_deployments import ChatCompletionDeployment
from tests.conftest import BASE_URL, DEFAULT_API_VERSION
from tests.utils import ai, assert_dialog, sys, user

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


def models_request_http() -> Any:
    response = requests.get(f"{BASE_URL}/openai/models")
    assert response.status_code == 200
    return response.json()


def models_request_openai() -> Any:
    return openai.Model.list(
        api_type="azure",
        api_base=BASE_URL,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )


def assert_models_subset(models: Any):
    actual_models = [model["id"] for model in models["data"]]
    expected_models = list(map(lambda e: e.value, deployments))

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"


def test_model_list_http(server):
    assert_models_subset(models_request_http())


def test_model_list_openai(server):
    assert_models_subset(models_request_openai())


@dataclass
class ValidationTestCase:
    name: str
    deployment: ChatCompletionDeployment
    messages: List[BaseMessage]
    expected_error: str

    def get_id(self) -> str:
        return f"{self.deployment.value}: {self.name}"


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


def get_validation_test_cases(
    deployment: ChatCompletionDeployment,
) -> List[ValidationTestCase]:
    return [
        ValidationTestCase(
            name="empty history",
            deployment=deployment,
            messages=[],
            expected_error=EMPTY_HISTORY_ERROR,
        ),
        ValidationTestCase(
            name="single system message",
            deployment=deployment,
            messages=[sys("Act as a helpful assistant")],
            expected_error=ONLY_SYS_MESSAGE_ERROR,
        ),
        ValidationTestCase(
            name="two system messages",
            deployment=deployment,
            messages=[
                sys("Act as a helpful assistant"),
                sys("Act as a tax accountant"),
                user("2+2=?"),
            ],
            expected_error=EXTRA_SYS_MESSAGE_ERROR,
        ),
        ValidationTestCase(
            name="single empty user message",
            deployment=deployment,
            messages=[user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
        ValidationTestCase(
            name="last empty user message",
            deployment=deployment,
            messages=[user("2+2=?"), ai("4"), user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
        ValidationTestCase(
            name="last message is not human",
            deployment=deployment,
            messages=[ai("5"), user("2+2=?"), ai("4")],
            expected_error=LAST_IS_NOT_HUMAN_ERROR,
        ),
        ValidationTestCase(
            name="three user messages in a row",
            deployment=deployment,
            messages=[user("2+3=?"), user("2+4=?"), user("2+5=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR,
        ),
        ValidationTestCase(
            name="two user messages in a row",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_LEN_ERROR,
        ),
        ValidationTestCase(
            name="ai then user",
            deployment=deployment,
            messages=[ai("5"), user("2+4=?"), user("2+4=?")],
            expected_error=INCORRECT_DIALOG_STRUCTURE_ROLES_ERROR,
        ),
    ]


validation_test_cases: List[ValidationTestCase] = [
    test_case
    for deployment in deployments
    for test_case in get_validation_test_cases(deployment)
] + [
    ValidationTestCase(
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
async def test_input_validation(server, test: ValidationTestCase):
    streaming = False
    model = create_chat_model(BASE_URL, test.deployment, streaming)

    with pytest.raises(Exception) as exc_info:
        await assert_dialog(model, test.messages, lambda s: True, streaming)

    assert isinstance(exc_info.value, openai.error.OpenAIError)
    assert exc_info.value.http_status == 422
    assert re.match(test.expected_error, str(exc_info.value))
