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

_CENTRAL = "us-central1"
_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.GEMINI_2_5_PRO: _CENTRAL,
    D.GEMINI_2_5_FLASH: _CENTRAL,
}


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


def _create_prompt(n: int) -> Tuple[str, Dict[int, int]]:
    lines = []
    answers = {}
    for idx in range(1, n + 1):
        x = _pseudo_random(2 * idx)
        y = _pseudo_random(2 * idx + 1)
        lines.append(f"[{idx}] {x} + {y} = ?")
        answers[idx] = x + y
    return "\n".join(lines), answers


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_implicit_caching(chat: Chat):
    message, answers = _create_prompt(300)

    messages: List[ChatCompletionMessageParam] = [sys(message)]

    indices = [151, 132, 267]
    for i, idx in enumerate(indices):
        query = f"Print the expression [{idx}] and compute it."
        answer = str(answers[idx])

        messages.append(user(query))

        response = await chat(messages=messages, max_tokens=512)
        assert answer in response.content

        messages.append(ai(response.content))

        assert response.usage is not None

        # Make sure that the prompt size is over the token threshold that
        # triggers the implicit caching:
        # https://ai.google.dev/gemini-api/docs/caching?lang=python#implicit-caching
        assert response.usage.prompt_tokens >= 4_096

        if i:
            assert (details := response.usage.prompt_tokens_details) is not None
            assert (cached := details.cached_tokens) is not None
            assert cached > 0
