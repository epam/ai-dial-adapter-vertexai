import io
from pathlib import Path
from typing import Callable, List, Mapping
from unittest.mock import patch

import openai
import pytest
from openai import AsyncAzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from PIL import Image

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.resource import Resource
from tests.integration_tests.constants import DOG_PICTURE
from tests.utils.exception import expected_exception
from tests.utils.mock_storage import MockFileStorage
from tests.utils.openai import chat_completion, user, user_with_image_url


@pytest.fixture(autouse=True)
def mock_storage(request):
    test_name = request.node.name
    root_dir = Path(__file__).parent / "mock-storage" / test_name
    with MockFileStorage.create(root_dir) as storage:
        with patch(
            "aidial_adapter_vertexai.adapters.create_file_storage",
            return_value=storage,
        ):
            yield storage


_CENTRAL = "us-central1"
_EAST = "us-east5"
_GLOBAL = "global"

_IMAGEN_MODELS: Mapping[D, str] = {
    D.IMAGEN_005: _CENTRAL,
    D.IMAGEN_3_GENERATE_001: _CENTRAL,
    D.IMAGEN_3_GENERATE_002: _CENTRAL,
    D.IMAGEN_3_FAST_GENERATE: _CENTRAL,
    D.IMAGEN_4_GENERATE_PREVIEW: _CENTRAL,
    D.IMAGEN_4_FAST_GENERATE_PREVIEW: _CENTRAL,
    D.IMAGEN_4_ULTRA_GENERATE_PREVIEW: _CENTRAL,
    D.IMAGEN_4_GENERATE: _CENTRAL,
    D.IMAGEN_4_FAST_GENERATE: _CENTRAL,
    D.IMAGEN_4_ULTRA_GENERATE: _CENTRAL,
}

_RETIRED_MODELS: Mapping[D, str] = {
    D.IMAGEN_005: _CENTRAL,
}

_LIVE_IMAGEN_MODELS: Mapping[D, str] = {
    k: v for k, v in _IMAGEN_MODELS.items() if k not in _RETIRED_MODELS
}

_GEMINI_IMAGE_MODELS: Mapping[D, str] = {
    D.GEMINI_2_0_FLASH_EXP: _GLOBAL,
    D.GEMINI_2_5_FLASH_IMAGE_PREVIEW: _GLOBAL,
    D.GEMINI_2_5_FLASH_IMAGE: _GLOBAL,
    D.GEMINI_3_PRO_IMAGE_PREVIEW: _GLOBAL,
    D.GEMINI_3_1_FLASH_IMAGE_PREVIEW: _GLOBAL,
}

_IMAGE_GENERATION_MODELS = _LIVE_IMAGEN_MODELS | _GEMINI_IMAGE_MODELS
_IMAGE_TO_IMAGE_MODELS = _GEMINI_IMAGE_MODELS
_VISION_MODEL = D.CLAUDE_4_SONNET


@pytest.fixture
def vision_model(get_openai_client: Callable[..., AsyncAzureOpenAI]):
    return get_openai_client(_VISION_MODEL.value, region=_EAST)


@pytest.mark.parametrize("deployment, region", _RETIRED_MODELS.items())
async def test_retired_models(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    async with expected_exception(
        cls=openai.NotFoundError,
        status_code=404,
        message="reached its end of life",
    ):
        client = get_openai_client(deployment.value, region=region)
        await client.chat.completions.create(
            model=deployment.value, messages=[user("test")]
        )


@pytest.mark.parametrize("deployment, region", _IMAGE_GENERATION_MODELS.items())
async def test_text_to_image_default(
    mock_storage: FileStorage,
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

    image_bytes = await _extract_image_bytes(mock_storage, imagen_response)

    vision_response = await vision_model.chat.completions.create(
        model=_VISION_MODEL.value,
        messages=[
            user_with_image_url(
                "What's the primary color of an object the animal is holding in its mouth? Answer with ONE word ONLY.",
                Resource(type="image/png", data=image_bytes),
            )
        ],
    )
    assert "red" in (vision_response.choices[0].message.content or "").lower()


@pytest.mark.parametrize("deployment, region", _GEMINI_IMAGE_MODELS.items())
async def test_text_to_image_aspect_ratio(
    mock_storage: FileStorage,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    configuration = {"image_config": {"aspect_ratio": "16:9"}}
    imagen_response = await client.chat.completions.create(
        model=deployment.value,
        messages=[user("a park in the spring next to a lake")],
        extra_body={"custom_fields": {"configuration": configuration}},
    )

    image_bytes = await _extract_image_bytes(mock_storage, imagen_response)

    with Image.open(io.BytesIO(image_bytes)) as img:
        w, h = img.size
        ratio = w / h
        assert (
            abs(ratio - 16 / 9) / ratio < 0.2
        ), f"Unexpected aspect ratio: {w=}/{h=} = {ratio:.2f}"


@pytest.mark.parametrize("deployment, region", _IMAGE_TO_IMAGE_MODELS.items())
async def test_edit_image_from_user(
    mock_storage: FileStorage,
    vision_model: AsyncAzureOpenAI,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    edit_response = await client.chat.completions.create(
        model=deployment.value,
        messages=[
            user_with_image_url(
                "modify the image by replacing the background with a forest meadow",
                DOG_PICTURE,
            )
        ],
    )

    edited_image = await _extract_image_bytes(mock_storage, edit_response)

    verification_prompt = (
        "Considering only the background (ignore the animal), which single word best "
        "describes it: forest, meadow, beach, city, indoor, desert, mountain, water, plain, other? "
        "Answer with ONE word ONLY."
    )
    vision_response = await vision_model.chat.completions.create(
        model=_VISION_MODEL.value,
        messages=[
            user_with_image_url(
                verification_prompt,
                Resource(type="image/png", data=edited_image),
            )
        ],
    )

    answer = (vision_response.choices[0].message.content or "").lower()
    assert any(w in answer for w in ("forest", "meadow"))


@pytest.mark.parametrize("deployment, region", _IMAGE_TO_IMAGE_MODELS.items())
async def test_edit_image_from_assistant(
    mock_storage: FileStorage,
    vision_model: AsyncAzureOpenAI,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    messages = []
    messages.append(user("generate a close up image of a siamese cat"))

    response1 = await client.chat.completions.create(
        model=deployment.value, messages=messages
    )

    assistant_message = response1.choices[0].message.model_dump(
        exclude_none=True
    )

    messages.append(assistant_message)
    messages.append(
        user("now put a fedora hat and tortoise shell glasses on the cat")
    )

    response2 = await client.chat.completions.create(
        model=deployment.value, messages=messages
    )

    image = await _extract_image_bytes(mock_storage, response2)

    verification_prompt = """
Which one of the following descriptions describes the given image best?
1. Crouching tiger
2. Cat with a hat and glasses
3. Picnic in a forest
4. Amazon river
Answer ONLY with the index of the best description as a single digit number.
DO NOT GENERATE IMAGES.
"""
    vision_response = await vision_model.chat.completions.create(
        model=_VISION_MODEL.value,
        messages=[
            user_with_image_url(
                verification_prompt,
                Resource(type="image/png", data=image),
            )
        ],
    )

    answer = (vision_response.choices[0].message.content or "").lower()
    assert "2" in answer
    assert all(w not in answer for w in ("1", "3", "4"))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("deployment, region", _LIVE_IMAGEN_MODELS.items())
async def test_content_filtering_imagen(
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
    assert resp["error"]["code"] == "content_filter"


async def _extract_image_bytes(
    storage: FileStorage, response: ChatCompletion
) -> bytes:
    assert len(response.choices) > 0
    choice = response.choices[0]

    assert choice.message.content is not None, "Message content is missing"
    cc = choice.message.custom_content  # type: ignore

    assert cc is not None, "Custom content is missing"

    for attachment in cc["attachments"]:
        if (
            attachment.get("title") == "Image"
            and attachment.get("type") == "image/png"
        ):
            if (url := attachment.get("url")) is not None:
                return await storage.download_file(url)

            if (data := attachment.get("data")) is not None:
                return data.encode()

            assert False, "Neither url nor data field is provided."

    assert False, "No image attachments were found"
