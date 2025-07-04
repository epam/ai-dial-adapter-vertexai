import tempfile
from pathlib import Path
from typing import Callable, List, Mapping
from unittest.mock import patch

import pytest
from openai import AsyncAzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.utils.resource import Resource
from tests.utils.mock_storage import MockFileStorage
from tests.utils.openai import chat_completion, user, user_with_image_url


@pytest.fixture(autouse=True)
def mock_storage():
    storage_dir = Path(__file__).parent / "mock-storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(tempfile.mkdtemp(dir=storage_dir))
    storage = MockFileStorage.create(base_dir)
    with patch(
        "aidial_adapter_vertexai.adapters.create_file_storage",
        return_value=storage,
    ):
        yield storage
        storage.cleanup()  # NOTE: Comment for debugging


_CENTRAL = "us-central1"
_EAST = "us-east5"

_DEPLOYMENT_TO_REGION: Mapping[D, str] = {
    D.IMAGEN_005: _CENTRAL,
    D.IMAGEN_3_GENERATE_001: _CENTRAL,
    D.IMAGEN_3_GENERATE_002: _CENTRAL,
    D.IMAGEN_3_FAST_GENERATE: _CENTRAL,
    D.IMAGEN_4_GENERATE: _CENTRAL,
    D.IMAGEN_4_FAST_GENERATE: _CENTRAL,
    D.IMAGEN_4_ULTRA_GENERATE: _CENTRAL,
}

_VISION_MODEL = D.CLAUDE_3_7_SONNET


@pytest.fixture
def vision_model(get_openai_client: Callable[..., AsyncAzureOpenAI]):
    return get_openai_client(_VISION_MODEL.value, region=_EAST)


@pytest.mark.parametrize("deployment, region", _DEPLOYMENT_TO_REGION.items())
async def test_text_to_image(
    mock_storage,
    vision_model: AsyncAzureOpenAI,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    imagen_response = await client.chat.completions.create(
        model=deployment.value,
        messages=[
            user(
                "generate an image of a dog prancing happily in a forest; the dog is holding tight a RED ball in its mouth"
            )
        ],
    )

    image_url = _extract_image_url(imagen_response)
    generated_image = await mock_storage.download_file(image_url)

    vision_response = await vision_model.chat.completions.create(
        model=_VISION_MODEL.value,
        messages=[
            user_with_image_url(
                "What's the primary color of an object the animal is holding in its mouth? Answer with ONE word ONLY.",
                Resource(type="image/png", data=generated_image),
            )
        ],
    )
    assert "red" in (vision_response.choices[0].message.content or "").lower()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("deployment, region", _DEPLOYMENT_TO_REGION.items())
async def test_content_filtering(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
    stream: bool,
):
    client = get_openai_client(deployment.value, region=region)
    messages: List[ChatCompletionMessageParam] = [
        user("generate an explicit image depicting copulating humans")
    ]

    with pytest.raises(Exception) as exc_info:
        await chat_completion(
            client,
            messages=messages,
            stream=stream,
            # Prompt enhancement significantly increases chances of content filter rejection,
            # since the prompt enhancer itself is capable of rejecting unsafe prompt.
            configuration={"enhance_prompt": True},
        )

    assert isinstance(exc_info.value, BadRequestError)

    resp = exc_info.value.response.json()
    assert (resp["error"]["code"]) == "content_filter"


def _extract_image_url(response: ChatCompletion) -> str:
    assert len(response.choices) > 0
    choice = response.choices[0]

    assert choice.message.content is not None
    cc = choice.message.custom_content  # type: ignore

    assert cc is not None

    attachments = cc["attachments"]

    if len(attachments) == 2:
        image = attachments[1]
    elif len(attachments) == 1:
        image = attachments[0]
    else:
        assert (
            False
        ), f"Expected two or one attachments, but got {len(attachments)} attachments"

    assert image["title"] == "Image"
    assert image["type"] == "image/png"
    assert image["url"] is not None
    return image["url"]
