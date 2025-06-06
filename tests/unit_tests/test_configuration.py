from typing import Mapping, Tuple

import httpx
import openai
import pytest

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import get_extra_headers
from tests.utils.openai import chat_completion, configuration, user

_EAST = "us-east5"

_chat_deployments: Mapping[ChatCompletionDeployment, str] = {
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2: _EAST,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU: _EAST,
    ChatCompletionDeployment.CLAUDE_3_OPUS: _EAST,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_3_HAIKU: _EAST,
    ChatCompletionDeployment.CLAUDE_3_7_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_4_SONNET: _EAST,
    ChatCompletionDeployment.CLAUDE_4_OPUS: _EAST,
}


async def _supports_citations(
    client: httpx.AsyncClient, region: str, deployment: ChatCompletionDeployment
) -> bool:
    conf = await configuration(
        client, deployment.value, get_extra_headers(region)
    )
    assert conf is not None
    return "enable_citations" in conf["properties"]


@pytest.mark.parametrize("test", _chat_deployments.items())
async def test_supports_citations(
    test_http_client: httpx.AsyncClient,
    test: Tuple[ChatCompletionDeployment, str],
):
    deployment, region = test
    assert await _supports_citations(test_http_client, region, deployment)


_invalid_configuration_test_cases = [
    (
        {"enable_citations": "hello"},
        "Invalid request. Path: 'custom_fields.configuration.enable_citations', error: value could not be parsed to a boolean",
    ),
    (
        {"extra_field": "extra value"},
        "Invalid request. Path: 'custom_fields.configuration.extra_field', error: extra fields not permitted",
    ),
]


@pytest.mark.parametrize("test", _invalid_configuration_test_cases)
@pytest.mark.parametrize("deployment", _chat_deployments.items())
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_configuration(
    get_openai_client,
    deployment: Tuple[ChatCompletionDeployment, str],
    stream: bool,
    test: Tuple[dict, str],
):
    deployment_enum, region = deployment
    deployment_id = deployment_enum.value
    client: openai.AsyncAzureOpenAI = get_openai_client(
        deployment_id, region=region
    )

    configuration, expected_error_message = test

    with pytest.raises(openai.APIStatusError) as exc:
        await chat_completion(
            client,
            messages=[user("test")],
            stream=stream,
            configuration=configuration,
        )

    assert exc.value.status_code == 422
    assert exc.value.body["message"] == expected_error_message  # type: ignore
