from pathlib import Path
from typing import Callable, Mapping
from unittest.mock import patch

import pytest
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from tests.utils.mock_storage import MockFileStorage
from tests.utils.openai import user


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

_VEO_MODELS: Mapping[D, str] = {
    D.VEO_3_0_GENERATE: _CENTRAL,
    D.VEO_3_0_GENERATE_PREVIEW: _CENTRAL,
    D.VEO_3_0_FAST_GENERATE: _CENTRAL,
    D.VEO_3_0_FAST_GENERATE_PREVIEW: _CENTRAL,
    D.VEO_3_1_GENERATE: _CENTRAL,
    D.VEO_3_1_GENERATE_PREVIEW: _CENTRAL,
    D.VEO_3_1_FAST_GENERATE: _CENTRAL,
    D.VEO_3_1_FAST_GENERATE_PREVIEW: _CENTRAL,
}


@pytest.mark.parametrize("deployment, region", _VEO_MODELS.items())
async def test_text_to_video(
    mock_storage: FileStorage,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: D,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    configuration = {"aspect_ratio": "16:9", "duration_seconds": 4}
    video_response = await client.chat.completions.create(
        model=deployment.value,
        messages=[user("generate a video of a Corgi dog")],
        extra_body={"custom_fields": {"configuration": configuration}},
    )

    video_bytes = await _extract_video_bytes(mock_storage, video_response)
    assert video_bytes is not None


async def _extract_video_bytes(
    storage: FileStorage, response: ChatCompletion
) -> bytes:
    assert len(response.choices) > 0
    choice = response.choices[0]

    assert choice.message.content is not None, "Message content is missing"
    cc = choice.message.custom_content  # type: ignore

    assert cc is not None, "Custom content is missing"

    for attachment in cc["attachments"]:
        if (
            attachment.get("title") == "Video"
            and attachment.get("type") == "video/mp4"
        ):
            if (url := attachment.get("url")) is not None:
                return await storage.download_file(url)

            if (data := attachment.get("data")) is not None:
                return data.encode()

            assert False, "Neither url nor data field is provided."

    assert False, "No video attachments were found"
