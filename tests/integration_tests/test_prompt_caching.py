import random
from typing import Awaitable, Callable, Dict, List, Mapping, Tuple, Unpack

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.utils.openai import (
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    chat_completion,
    sanitize_test_name,
    sys,
    user,
)
from tests.utils.selector import Selector

_CENTRAL = "us-central1"
_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.GEMINI_2_0_FLASH_EXP: _CENTRAL,
    D.GEMINI_2_0_FLASH_001: _CENTRAL,
    D.GEMINI_2_5_PRO: _CENTRAL,
    D.GEMINI_2_0_FLASH_LITE_1: _CENTRAL,
    D.GEMINI_2_5_FLASH: _CENTRAL,
}


def select(p: Selector[D], xs: List[D]) -> List[D]:
    return [x for x in xs if p(x)]


deployments = list(_DEPLOYMENT_TO_REGION.keys())


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


def _pseudo_random(seed: int, a: int = 0, b: int = 100) -> int:
    return random.Random(seed).randrange(a, b + 1)


def _create_large_prompt(n: int) -> Tuple[str, Dict[int, int]]:
    lines = []
    answers = {}
    for idx in range(1, n + 1):
        x = _pseudo_random(2 * idx)
        y = _pseudo_random(2 * idx + 1)
        lines.append(f"[{idx}] {x} + {y} = ?")
        answers[idx] = x + y
    return "\n".join(lines), answers


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_caching(chat: Chat):
    message, answers = _create_large_prompt(300)

    messages: List[ChatCompletionMessageParam] = [sys(message)]

    indices = [151, 132, 267]
    for idx in indices:
        query = (
            f"Compute the expression number {idx}. Reply with a single number."
        )
        answer = str(answers[idx])

        messages.append(user(query))

        response = await chat(messages=messages, max_tokens=50)
        assert answer in response.content

        messages.append(ai(response.content))

        assert response.usage is not None
        assert response.usage.prompt_tokens >= 4_096
