from typing import List, Tuple

import pytest
import requests

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from tests.conftest import TEST_SERVER_URL

test_cases: List[Tuple[ChatCompletionDeployment, bool, bool]] = [
    (ChatCompletionDeployment.CHAT_BISON_1, True, True),
    (ChatCompletionDeployment.CHAT_BISON_2, True, True),
    (ChatCompletionDeployment.CHAT_BISON_2_32K, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_1, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_2, True, True),
    (ChatCompletionDeployment.CODECHAT_BISON_2_32K, True, True),
    (ChatCompletionDeployment.GEMINI_PRO_1, True, False),
    (ChatCompletionDeployment.GEMINI_PRO_VISION_1, True, False),
    (ChatCompletionDeployment.GEMINI_PRO_VISION_1_5, True, False),
    (ChatCompletionDeployment.IMAGEN_005, True, True),
]


def feature_test_helper(
    url: str, is_supported: bool, headers: dict, payload: dict
) -> None:
    response = requests.post(url, json=payload, headers=headers)
    assert (
        response.status_code != 404
    ) == is_supported, (
        f"is_supported={is_supported}, code={response.status_code}, url={url}"
    )


@pytest.mark.parametrize(
    "deployment, tokenize_supported, truncate_supported", test_cases
)
def test_model_features(
    server,
    deployment: ChatCompletionDeployment,
    tokenize_supported: bool,
    truncate_supported: bool,
):
    payload = {"inputs": []}
    headers = {"Content-Type": "application/json", "Api-Key": "dummy"}

    BASE_URL = f"{TEST_SERVER_URL}/openai/deployments/{deployment.value}"

    tokenize_url = f"{BASE_URL}/tokenize"
    feature_test_helper(tokenize_url, tokenize_supported, headers, payload)

    truncate_url = f"{BASE_URL}/truncate_prompt"
    feature_test_helper(truncate_url, truncate_supported, headers, payload)
