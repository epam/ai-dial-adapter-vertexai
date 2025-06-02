import re
from dataclasses import dataclass
from typing import List

import pytest
from openai import BadRequestError, UnprocessableEntityError
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.utils.resource import Resource
from tests.utils.openai import (
    ChatCompletionResult,
    ai,
    chat_completion,
    sanitize_test_name,
    sys,
    user,
    user_with_attachment_data,
)
from tests.utils.pdf import gen_pdf

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
EXTRA_SYS_MESSAGE_ERROR = "System and developer messages other than the initial system message are not allowed"
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


@pytest.mark.parametrize(
    "test", validation_test_cases, ids=lambda test: test.get_id()
)
async def test_input_validation(get_openai_client, test: TestCase):
    client = get_openai_client(test.deployment.value)

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(client, test.messages, stream=False)

    if test.expected_exception is not None:
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        assert isinstance(exc_info.value, UnprocessableEntityError)
        assert re.search(str(test.expected_exception), str(exc_info.value))
    else:
        await run_chat_completion()


async def test_imagen_content_filtering(get_openai_client):
    client = get_openai_client(ChatCompletionDeployment.IMAGEN_005.value)
    messages: List[ChatCompletionMessageParam] = [
        user("generate something unsafe")
    ]

    with pytest.raises(Exception) as exc_info:
        await chat_completion(client, messages, stream=False)

    assert isinstance(exc_info.value, BadRequestError)

    resp = exc_info.value.response.json()
    assert (resp["error"]["code"]) == "content_filter"
    assert (
        resp["error"]["message"]
        == "The response is blocked, as it may violate our policies."
    )


async def test_gemini_pdf_page_overflow_for_document(get_openai_client):
    client = get_openai_client(ChatCompletionDeployment.GEMINI_PRO_1_5_V2.value)

    doc = Resource(type="application/pdf", data=gen_pdf(["a"] * 2_000))

    messages: List[ChatCompletionMessageParam] = [
        user_with_attachment_data("test", doc)
    ]

    with pytest.raises(Exception) as exc_info:
        await chat_completion(client, messages, stream=False)

    assert isinstance(exc_info.value, UnprocessableEntityError)

    error = exc_info.value.response.json()["error"]
    expected_message = "The following files failed to process:\n1. data attachment: the number of pages in the document (2000) exceeds the limit (1000)"
    assert error["message"] == expected_message
    assert error["display_message"] == expected_message


async def test_gemini_pdf_page_overflow_for_request(get_openai_client):
    client = get_openai_client(ChatCompletionDeployment.GEMINI_PRO_1_5_V2.value)

    doc = Resource(type="application/pdf", data=gen_pdf(["a"] * 1_000))

    messages: List[ChatCompletionMessageParam] = [
        user_with_attachment_data("test", doc, doc, doc, doc)
    ]

    with pytest.raises(Exception) as exc_info:
        await chat_completion(client, messages, stream=False)

    assert isinstance(exc_info.value, UnprocessableEntityError)

    error = exc_info.value.response.json()["error"]
    expected_message = (
        "The total number of pages in PDF documents exceeds the limit (3000)"
    )
    assert error["message"] == expected_message
    assert error["display_message"] == expected_message
