from dataclasses import dataclass
from typing import List

import httpx
import pytest

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.utils.openai import sanitize_test_name


@dataclass
class TestCase:
    __test__ = False

    deployment: D
    tokenize_supported: bool
    truncate_supported: bool
    configuration_supported: bool

    def get_id(self):
        return sanitize_test_name(self.deployment.value)


test_cases: List[TestCase] = [
    TestCase(D.CHAT_BISON_1, True, True, False),
    TestCase(D.CHAT_BISON_2, True, True, False),
    TestCase(D.CHAT_BISON_2_32K, True, True, False),
    TestCase(D.CODECHAT_BISON_1, True, True, False),
    TestCase(D.CODECHAT_BISON_2, True, True, False),
    TestCase(D.CODECHAT_BISON_2_32K, True, True, False),
    TestCase(D.GEMINI_PRO_1, True, True, False),
    TestCase(D.GEMINI_PRO_VISION_1, True, True, False),
    TestCase(D.GEMINI_PRO_1_5_PREVIEW, True, True, False),
    TestCase(D.GEMINI_PRO_1_5_V1, True, True, False),
    TestCase(D.GEMINI_PRO_1_5_V2, True, True, False),
    TestCase(D.GEMINI_FLASH_1_5_V1, True, True, False),
    TestCase(D.GEMINI_FLASH_1_5_V2, True, True, False),
    TestCase(D.IMAGEN_005, True, True, True),
    TestCase(D.GEMINI_2_0_FLASH_EXP, True, True, False),
    TestCase(D.GEMINI_2_0_FLASH_001, True, True, False),
    TestCase(D.GEMINI_2_0_FLASH_THINKING_EXP_01_21, True, True, False),
    TestCase(D.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05, True, True, False),
    TestCase(D.GEMINI_2_5_PRO, True, True, True),
    TestCase(D.GEMINI_2_0_PRO_EXP_02_05, True, True, False),
    TestCase(D.GEMINI_2_5_PRO_EXP_03_25, True, True, True),
    TestCase(D.GEMINI_2_5_PRO_PREVIEW_03_25, True, True, True),
    TestCase(D.GEMINI_2_0_FLASH_LITE_1, True, True, False),
    TestCase(D.GEMINI_2_5_FLASH, True, True, True),
    TestCase(D.GEMINI_2_5_FLASH_PREVIEW_04_17, True, True, True),
    TestCase(D.CLAUDE_3_5_SONNET_V2, True, True, True),
    TestCase(D.CLAUDE_3_5_HAIKU, True, True, True),
    TestCase(D.CLAUDE_3_OPUS, True, True, True),
    TestCase(D.CLAUDE_3_5_SONNET, True, True, True),
    TestCase(D.CLAUDE_3_HAIKU, True, True, True),
    TestCase(D.CLAUDE_3_7_SONNET, True, True, True),
    TestCase(D.CLAUDE_4_OPUS, True, True, True),
    TestCase(D.CLAUDE_4_SONNET, True, True, True),
]


async def assert_feature(
    http_client: httpx.AsyncClient,
    endpoint: str,
    is_supported: bool,
    headers: dict,
    payload: dict | None,
) -> None:
    if payload is None:
        response = await http_client.get(endpoint, headers=headers)
    else:
        response = await http_client.post(
            endpoint, json=payload, headers=headers
        )

    if (response.status_code == 200 and not is_supported) or (
        response.status_code == 404 and is_supported
    ):
        assert (
            False
        ), f"is_supported={is_supported}, code={response.status_code}, url={endpoint}"


@pytest.mark.parametrize("test", test_cases, ids=lambda test: test.get_id())
async def test_model_features(
    test_http_client: httpx.AsyncClient, test: TestCase
):
    payload = {"inputs": []}
    headers = {"Content-Type": "application/json", "Api-Key": "dummy"}

    base = f"openai/deployments/{test.deployment.value}"

    tokenize_endpoint = f"{base}/tokenize"
    await assert_feature(
        test_http_client,
        tokenize_endpoint,
        test.tokenize_supported,
        headers,
        payload,
    )

    truncate_endpoint = f"{base}/truncate_prompt"
    await assert_feature(
        test_http_client,
        truncate_endpoint,
        test.truncate_supported,
        headers,
        payload,
    )

    configuration_endpoint = f"{base}/configuration"
    await assert_feature(
        test_http_client,
        configuration_endpoint,
        test.configuration_supported,
        headers,
        None,
    )
