from dataclasses import dataclass
from typing import Callable, List

import pytest
from langchain.schema import BaseMessage

from client.client_adapter import create_chat_model
from llm.vertex_ai_deployments import ChatCompletionDeployment
from tests.conftest import BASE_URL
from tests.utils import assert_dialog, sanitize_test_name, user

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


@dataclass
class TestCase:
    __test__ = False

    deployment: ChatCompletionDeployment
    streaming: bool

    query: str | List[BaseMessage]
    test: Callable[[str], bool]

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {self.query}"
        )

    def get_messages(self) -> List[BaseMessage]:
        return [user(self.query)] if isinstance(self.query, str) else self.query


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            deployment=deployment,
            streaming=streaming,
            query="2+3=?",
            test=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            deployment=deployment,
            streaming=streaming,
            query='Reply with "Hello"',
            test=lambda s: "hello" in s.lower(),
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
    model = create_chat_model(BASE_URL, test.deployment, streaming)
    await assert_dialog(model, test.get_messages(), test.test, streaming)
