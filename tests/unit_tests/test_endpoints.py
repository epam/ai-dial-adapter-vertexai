from typing import List, Tuple

import httpx
import pytest

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment

test_cases: List[Tuple[ChatCompletionDeployment, bool, bool]] = [
    (ChatCompletionDeployment.CHAT_BISON_1, True, True),
    (ChatCompletionDeployment.CHAT_BISON_2, True, True),
    (ChatCompletionDeployment.CHAT_BISON_2_32K, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_1, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_2, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_2_32K, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_1, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_VISION_1, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_1_5_V1, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_1_5_V2, True, True),
    (ChatCompletionDeployment.GEMINI_FLASH_1_5_V1, True, True),
    (ChatCompletionDeployment.GEMINI_FLASH_1_5_V2, True, True),
    (ChatCompletionDeployment.IMAGEN_005, True, True),
]


async def assert_feature(
    test_http_client: httpx.AsyncClient,
    endpoint: str,
    is_supported: bool,
    headers: dict,
    payload: dict,
) -> None:
    response = await test_http_client.post(
        endpoint, json=payload, headers=headers
    )
    assert (
        response.status_code != 404
    ) == is_supported, f"is_supported={is_supported}, code={response.status_code}, url={endpoint}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment, tokenize_supported, truncate_supported", test_cases
)
async def test_model_features(
    test_http_client: httpx.AsyncClient,
    deployment: ChatCompletionDeployment,
    tokenize_supported: bool,
    truncate_supported: bool,
):
    payload = {"inputs": []}
    headers = {"Content-Type": "application/json", "Api-Key": "dummy"}

    base = f"openai/deployments/{deployment.value}"

    tokenize_endpoint = f"{base}/tokenize"
    await assert_feature(
        test_http_client,
        tokenize_endpoint,
        tokenize_supported,
        headers,
        payload,
    )

    truncate_endpoint = f"{base}/truncate_prompt"
    await assert_feature(
        test_http_client,
        truncate_endpoint,
        truncate_supported,
        headers,
        payload,
    )
