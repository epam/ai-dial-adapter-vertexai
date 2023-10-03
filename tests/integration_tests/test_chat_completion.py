from dataclasses import dataclass
from typing import Callable, List

import pytest
from langchain.schema import BaseMessage

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from client.client_adapter import create_chat_model
from tests.conftest import TEST_SERVER_URL
from tests.utils import assert_dialog, sanitize_test_name, sys, user

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment
    streaming: bool

    messages: List[BaseMessage]
    test: Callable[[str], bool]

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {self.name}"
        )


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            name="2+3=5",
            deployment=deployment,
            streaming=streaming,
            messages=[user("2+3=?")],
            test=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            name="hello",
            deployment=deployment,
            streaming=streaming,
            messages=[user('Reply with "Hello"')],
            test=lambda s: "hello" in s.lower(),
        )
    )

    ret.append(
        TestCase(
            name="empty sys message",
            deployment=deployment,
            streaming=streaming,
            messages=[sys(""), user("2+4=?")],
            test=lambda s: "6" in s,
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test_case
        for model in deployments
        for streaming in [False, True]
        for test_case in get_test_cases(model, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    streaming = test.streaming
    model = create_chat_model(TEST_SERVER_URL, test.deployment, streaming)
    await assert_dialog(model, test.messages, test.test, streaming)
