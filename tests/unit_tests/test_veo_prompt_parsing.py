from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.veo.prompt import VeoPromptParser
from aidial_adapter_vertexai.utils.resource import Resource


def _data_url(content_type: str, payload: bytes) -> str:
    return Resource(type=content_type, data=payload).to_data_url()


async def test_veo_prompt_parses_image_from_content_part():
    image_url = _data_url("image/png", b"image-bytes")
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Animate this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    )

    prompt = await VeoPromptParser.parse(
        file_storage=None,
        messages=[message],
    )

    assert prompt.text == "Animate this image"
    assert prompt.image is not None
    assert prompt.image.image_bytes == b"image-bytes"
    assert prompt.image.mime_type == "image/png"
    assert prompt.video is None


async def test_veo_prompt_parses_image_from_custom_attachment():
    image_url = _data_url("image/png", b"image-bytes")
    message = Message.model_validate(
        {
            "role": "user",
            "content": "Animate this image",
            "custom_content": {
                "attachments": [
                    {
                        "type": "image/png",
                        "url": image_url,
                    }
                ]
            },
        }
    )

    prompt = await VeoPromptParser.parse(
        file_storage=None,
        messages=[message],
    )

    assert prompt.text == "Animate this image"
    assert prompt.image is not None
    assert prompt.image.image_bytes == b"image-bytes"
    assert prompt.image.mime_type == "image/png"
    assert prompt.video is None


async def test_veo_prompt_parses_video_from_custom_attachment():
    video_url = _data_url("video/mp4", b"video-bytes")
    message = Message.model_validate(
        {
            "role": "user",
            "content": "Add more details to this video",
            "custom_content": {
                "attachments": [
                    {
                        "type": "video/mp4",
                        "url": video_url,
                    }
                ]
            },
        }
    )

    prompt = await VeoPromptParser.parse(
        file_storage=None,
        messages=[message],
    )

    assert prompt.text == "Add more details to this video"
    assert prompt.image is None
    assert prompt.video is not None
    assert prompt.video.video_bytes == b"video-bytes"
    assert prompt.video.mime_type == "video/mp4"
