from typing import Mapping, Tuple

import httpx
import openai
import pytest

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.conftest import get_extra_headers
from tests.utils.openai import chat_completion, configuration, user

_EAST = "us-east5"
_CENTRAL = "us-central1"

_claude_deployments: Mapping[D, str] = {
    D.CLAUDE_3_5_SONNET_V2: _EAST,
    D.CLAUDE_3_5_HAIKU: _EAST,
    D.CLAUDE_3_OPUS: _EAST,
    D.CLAUDE_3_5_SONNET: _EAST,
    D.CLAUDE_3_HAIKU: _EAST,
    D.CLAUDE_3_7_SONNET: _EAST,
    D.CLAUDE_4_SONNET: _EAST,
    D.CLAUDE_4_OPUS: _EAST,
    D.CLAUDE_4_1_OPUS: _EAST,
    D.CLAUDE_4_5_HAIKU: _EAST,
    D.CLAUDE_4_5_SONNET: _EAST,
}

_gemini_2_5_deployments: Mapping[D, str] = {
    D.GEMINI_2_5_PRO: _CENTRAL,
    D.GEMINI_2_5_FLASH: _CENTRAL,
    D.GEMINI_2_5_FLASH_IMAGE_PREVIEW: _CENTRAL,
}

_gemini_2_5_image_deployments: Mapping[D, str] = {
    D.GEMINI_2_5_FLASH_IMAGE_PREVIEW: _CENTRAL,
}


async def _configuration_has_field(
    client: httpx.AsyncClient, region: str, deployment: D, field_name: str
) -> bool:
    conf = await configuration(
        client, deployment.value, get_extra_headers(region)
    )
    assert conf is not None
    return field_name in conf["properties"]


@pytest.mark.parametrize("test", _gemini_2_5_deployments.items())
async def test_gemini_2_5_supports_thinking(
    test_http_client: httpx.AsyncClient, test: Tuple[D, str]
):
    deployment, region = test
    assert await _configuration_has_field(
        test_http_client, region, deployment, "thinking"
    )


@pytest.mark.parametrize("test", _gemini_2_5_image_deployments.items())
async def test_gemini_2_5_image_supports_image_config(
    test_http_client: httpx.AsyncClient, test: Tuple[D, str]
):
    deployment, region = test
    assert await _configuration_has_field(
        test_http_client, region, deployment, "image_config"
    )


@pytest.mark.parametrize("test", _claude_deployments.items())
async def test_claude_supports_citations(
    test_http_client: httpx.AsyncClient, test: Tuple[D, str]
):
    deployment, region = test
    assert await _configuration_has_field(
        test_http_client, region, deployment, "enable_citations"
    )


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
@pytest.mark.parametrize("deployment", _claude_deployments.items())
@pytest.mark.parametrize("stream", [False, True])
async def test_claude_invalid_configuration(
    get_openai_client,
    deployment: Tuple[D, str],
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
