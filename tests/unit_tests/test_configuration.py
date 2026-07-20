from dataclasses import dataclass

import httpx
import openai
import pytest
from pydantic import ValidationError

from aidial_adapter_vertexai.chat.gemini.adapter import GeminiConfigurationModel
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.conftest import get_extra_headers
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.openai import chat_completion, configuration, user

_claude_deployments: list[D] = [
    D.CLAUDE_3_5_SONNET_V2,
    D.CLAUDE_3_5_HAIKU,
    D.CLAUDE_3_OPUS,
    D.CLAUDE_3_5_SONNET,
    D.CLAUDE_3_HAIKU,
    D.CLAUDE_3_7_SONNET,
    D.CLAUDE_4_SONNET,
    D.CLAUDE_4_OPUS,
    D.CLAUDE_4_1_OPUS,
    D.CLAUDE_4_5_HAIKU,
    D.CLAUDE_4_5_SONNET,
    D.CLAUDE_4_6_SONNET,
    D.CLAUDE_4_6_OPUS,
    D.CLAUDE_4_5_OPUS,
]

_gemini_deployments_with_thinking: list[D] = [
    D.GEMINI_2_5_PRO,
    D.GEMINI_3_PRO,
    D.GEMINI_3_PRO_PREVIEW,
    D.GEMINI_3_1_PRO_PREVIEW,
    D.GEMINI_3_FLASH_PREVIEW,
]

_gemini_deployments_with_imagen: list[D] = [
    D.GEMINI_2_0_FLASH_EXP,
    D.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
    D.GEMINI_2_5_FLASH_IMAGE,
    D.GEMINI_3_PRO_IMAGE_PREVIEW,
    D.GEMINI_3_1_FLASH_IMAGE_PREVIEW,
]

_veo_deployments: list[D] = [
    D.VEO_3_0_GENERATE,
    D.VEO_3_0_GENERATE,
    D.VEO_3_0_GENERATE_PREVIEW,
    D.VEO_3_0_FAST_GENERATE,
    D.VEO_3_0_FAST_GENERATE_PREVIEW,
    D.VEO_3_1_GENERATE,
    D.VEO_3_1_GENERATE_PREVIEW,
    D.VEO_3_1_FAST_GENERATE,
    D.VEO_3_1_FAST_GENERATE_PREVIEW,
]


async def _configuration_has_field(
    client: httpx.AsyncClient, deployment: D, field_name: str
) -> bool:
    conf = await configuration(
        client, deployment.value, get_extra_headers("test-region")
    )
    assert conf is not None
    return field_name in conf["properties"]


@pytest.mark.parametrize("deployment", _gemini_deployments_with_thinking)
async def test_gemini_supports_thinking(
    test_http_client: httpx.AsyncClient, deployment: D
):
    assert await _configuration_has_field(
        test_http_client, deployment, "thinking"
    )


@pytest.mark.parametrize("deployment", _gemini_deployments_with_imagen)
async def test_gemini_supports_image_config(
    test_http_client: httpx.AsyncClient, deployment: D
):
    assert await _configuration_has_field(
        test_http_client, deployment, "image_config"
    )


@pytest.mark.parametrize(
    "deployment",
    _gemini_deployments_with_thinking + _gemini_deployments_with_imagen,
)
async def test_gemini_supports_safety_settings(
    test_http_client: httpx.AsyncClient, deployment: D
):
    assert await _configuration_has_field(
        test_http_client, deployment, "safety_settings"
    )


@pytest.mark.parametrize("deployment", _claude_deployments)
async def test_claude_supports_citations(
    test_http_client: httpx.AsyncClient, deployment: D
):
    assert await _configuration_has_field(
        test_http_client, deployment, "enable_citations"
    )


def test_gemini_safety_settings_must_be_a_list():
    with pytest.raises(ValidationError) as exc_info:
        GeminiConfigurationModel.model_validate(
            {
                "safety_settings": {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                }
            }
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("safety_settings",)
    assert errors[0]["type"] == "list_type"
    assert errors[0]["msg"] == "Input should be a valid list"


def test_gemini_safety_settings_reject_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        GeminiConfigurationModel.model_validate(
            {
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                        "extra_field": "value",
                    }
                ]
            }
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("safety_settings", 0, "extra_field")
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["msg"] == "Extra inputs are not permitted"


@dataclass
class ConfTestCase:
    configuration: dict
    exception: ExpectedException


_invalid_claude_configuration_test_cases: list[ConfTestCase] = [
    ConfTestCase(
        {"enable_citations": "hello"},
        ExpectedException(
            status_code=422,
            type=openai.APIStatusError,
            message="Invalid request. Path: 'custom_fields.configuration.enable_citations', error: Input should be a valid boolean, unable to interpret input",
        ),
    ),
    ConfTestCase(
        {"extra_field": "extra value"},
        ExpectedException(
            status_code=422,
            type=openai.APIStatusError,
            message="Invalid request. Path: 'custom_fields.configuration.extra_field', error: Extra inputs are not permitted",
        ),
    ),
]

_invalid_veo_configuration_test_cases = [
    ConfTestCase(
        {"n_variants": 10},
        ExpectedException(
            status_code=400,
            type=openai.APIStatusError,
            message="1 validation error for _GenerateVideosParameters.+config.n_variants.+Extra inputs are not permitted",
        ),
    ),
]


_invalid_gemini_configuration_test_cases = [
    ConfTestCase(
        {"thinking": {"budget_tokens": 1024}},
        ExpectedException(
            status_code=400,
            type=openai.APIStatusError,
            message="1 validation error for GenerateContentConfig.+thinking_config.budget_tokens.+Extra inputs are not permitted",
        ),
    ),
]


@pytest.mark.parametrize("test", _invalid_claude_configuration_test_cases)
@pytest.mark.parametrize("deployment", _claude_deployments)
@pytest.mark.parametrize("stream", [False, True])
async def test_claude_invalid_configuration(
    get_openai_client, deployment: D, stream: bool, test: ConfTestCase
):
    await _test_configuration(get_openai_client, deployment, stream, test)


@pytest.mark.parametrize("test", _invalid_veo_configuration_test_cases)
@pytest.mark.parametrize("deployment", _veo_deployments)
@pytest.mark.parametrize("stream", [False, True])
async def test_veo_invalid_configuration(
    get_openai_client, deployment: D, stream: bool, test: ConfTestCase
):
    await _test_configuration(get_openai_client, deployment, stream, test)


@pytest.mark.parametrize("test", _invalid_gemini_configuration_test_cases)
@pytest.mark.parametrize("deployment", _gemini_deployments_with_thinking)
@pytest.mark.parametrize("stream", [False, True])
async def test_gemini_invalid_configuration(
    get_openai_client, deployment: D, stream: bool, test: ConfTestCase
):
    await _test_configuration(get_openai_client, deployment, stream, test)


async def _test_configuration(
    get_openai_client, deployment: D, stream: bool, test: ConfTestCase
):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        deployment.value, region="test-region"
    )

    async with expected_exception(test.exception):
        await chat_completion(
            client,
            messages=[user("test")],
            stream=stream,
            configuration=test.configuration,
        )
