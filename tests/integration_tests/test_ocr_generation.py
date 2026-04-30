from collections.abc import Callable, Mapping

import pytest
from openai import AsyncAzureOpenAI

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.integration_tests.constants import OCR_PNG_RESOURCE
from tests.utils.openai import user_with_attachment_url, user_with_image_url

pytestmark = pytest.mark.slow

_CENTRAL = "us-central1"

_OCR_MODELS: Mapping[D, str] = {D.MISTRAL_OCR: _CENTRAL}

ALL_OCR_GEN_MODELS = _OCR_MODELS


@pytest.mark.parametrize("deployment, region", _OCR_MODELS.items())
async def test_ocr_document_from_attachment(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    messages = [
        user_with_attachment_url(
            "Find elements inside this paper.",
            OCR_PNG_RESOURCE,
        )
    ]
    response = await client.chat.completions.create(
        model=deployment.value,
        messages=messages,
    )

    content = (response.choices[0].message.content or "").lower()
    success_markers = ("nasdaq", "stocks", "usa", "money")
    assert any(marker in content for marker in success_markers), (
        f"Expected one of {success_markers} in OCR response, got: {content!r}"
    )


@pytest.mark.parametrize("deployment, region", _OCR_MODELS.items())
async def test_ocr_document_from_image_url(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    messages = [
        user_with_image_url(
            "Find elements inside this paper.",
            OCR_PNG_RESOURCE,
        )
    ]
    response = await client.chat.completions.create(
        model=deployment.value,
        messages=messages,
    )

    content = (response.choices[0].message.content or "").lower()
    success_markers = ("nasdaq", "stocks", "usa", "money")
    assert any(marker in content for marker in success_markers), (
        f"Expected one of {success_markers} in OCR response, got: {content!r}"
    )
