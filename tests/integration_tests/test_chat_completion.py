import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import TEST_SERVER_URL
from tests.utils.llm import (
    assert_dialog,
    create_chat_model,
    for_all,
    sanitize_test_name,
    sys,
    user,
)

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
    ChatCompletionDeployment.GEMINI_PRO_1,
]


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment
    streaming: bool

    messages: List[BaseMessage]
    expected: Callable[[List[str]], bool] | Exception

    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    n: Optional[int] = None

    def get_id(self):
        max_tokens_str = (
            f"max_tokens={self.max_tokens}"
            if self.max_tokens is not None
            else ""
        )
        stop_sequence_str = f"stop={self.stop}" if self.stop is not None else ""
        streaming_str = "streaming" if self.streaming else "non-streaming"
        n_str = f"n={self.n}" if self.n is not None else ""
        segments = [
            self.deployment.value,
            streaming_str,
            max_tokens_str,
            stop_sequence_str,
            n_str,
            self.name,
        ]
        return sanitize_test_name(" ".join(s for s in segments if s))


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    is_codechat = "codechat" in deployment.value

    ret: List[TestCase] = []

    ret.append(
        TestCase(
            name="2+3=5",
            deployment=deployment,
            streaming=streaming,
            messages=[user("2+3=?")],
            expected=for_all(lambda s: "5" in s),
        )
    )

    ret.append(
        TestCase(
            name="hello",
            deployment=deployment,
            streaming=streaming,
            messages=[user('Reply with "Hello"')],
            expected=for_all(lambda s: "hello" in s.lower()),
        )
    )

    ret.append(
        TestCase(
            name="empty sys message",
            deployment=deployment,
            streaming=streaming,
            messages=[sys(""), user("2+4=?")],
            expected=for_all(lambda s: "6" in s),
        )
    )

    ret.append(
        TestCase(
            name="non empty sys message",
            deployment=deployment,
            streaming=streaming,
            messages=[sys("Act as helpful assistant"), user("2+5=?")],
            expected=for_all(lambda s: "7" in s),
        )
    )

    ret.append(
        TestCase(
            name="max tokens 1",
            deployment=deployment,
            streaming=streaming,
            max_tokens=1,
            messages=[user("tell me the full story of Pinocchio")],
            expected=for_all(lambda s: len(s.split()) == 1),
        )
    )

    ret.append(
        TestCase(
            name="multiple candidates",
            deployment=deployment,
            streaming=streaming,
            max_tokens=10,
            n=5,
            messages=[user("heads or tails?")],
            expected=(
                Exception("n>1 is not supported in streaming mode")
                if streaming
                else for_all(lambda _: True, 5)
            ),
        )
    )
    ret.append(
        TestCase(
            name="stop sequence",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=["world"],
            messages=[user('Reply with "hello world"')],
            expected=(
                Exception(
                    "stop sequences are not supported for code chat model"
                )
                if is_codechat
                else for_all(lambda s: "world" not in s.lower())
            ),
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
    model = create_chat_model(
        TEST_SERVER_URL,
        test.deployment,
        test.streaming,
        test.max_tokens,
    )

    if isinstance(test.expected, Exception):
        with pytest.raises(Exception) as exc_info:
            await assert_dialog(
                model=model,
                messages=test.messages,
                output_predicate=lambda s: True,
                streaming=test.streaming,
                stop=test.stop,
                n=test.n,
            )

        assert isinstance(exc_info.value, openai.error.OpenAIError)
        assert exc_info.value.http_status == 422
        assert re.search(str(test.expected), str(exc_info.value))
    else:
        await assert_dialog(
            model=model,
            messages=test.messages,
            output_predicate=test.expected,
            streaming=test.streaming,
            stop=test.stop,
            n=test.n,
        )
