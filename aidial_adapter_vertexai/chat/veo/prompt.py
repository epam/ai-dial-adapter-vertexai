from dataclasses import dataclass
from typing import assert_never

from aidial_sdk.chat_completion import (
    Message,
    MessageContentAudioPart,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion.request import MessageContentRefusalPart
from google.genai.types import Image, Video

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.resource import (
    AttachmentResource,
    URLResource,
)
from aidial_adapter_vertexai.dial_api.storage import FileStorage


@dataclass
class VeoPrompt:
    text: str
    image: Image | None = None
    video: Video | None = None


class VeoPromptParser:
    @classmethod
    async def parse(
        cls,
        file_storage: FileStorage | None,
        messages: list[Message],
    ) -> VeoPrompt:
        if len(messages) == 0:
            raise ValidationError("The list of messages must not be empty")

        prompt_text = ""
        image: Image | None = None
        video: Video | None = None
        last_message = messages[-1]
        content = last_message.content

        match content:
            case None:
                pass
            case str():
                prompt_text = content
            case list():
                text_parts: list[str] = []
                for part in content:
                    match part:
                        case MessageContentTextPart(text=text):
                            text_parts.append(text)
                        case MessageContentImagePart(image_url=image_url):
                            resource = await URLResource(
                                url=image_url.url,
                                entity_name="image content part",
                            ).download(file_storage)
                            image = Image(
                                image_bytes=resource.data,
                                mime_type=resource.type,
                            )
                        case MessageContentFilePart():
                            raise ValidationError(
                                "File content parts aren't supported. Use attachments instead."
                            )
                        case MessageContentAudioPart():
                            raise ValidationError(
                                "Veo models don't support audio content parts"
                            )
                        case MessageContentRefusalPart():
                            raise ValidationError(
                                "Can't extract text from a refusal content part"
                            )
                        case _:
                            assert_never(part)
                prompt_text = "\n\n".join(text_parts)
            case _:
                assert_never(content)

        for attachment in get_attachments(last_message):
            attachment_resource = AttachmentResource(
                attachment=attachment, entity_name="image attachment"
            )
            content_type = await attachment_resource.guess_content_type()
            if content_type is None:
                continue

            resource = await attachment_resource.download(file_storage)
            if content_type.startswith("image/"):
                image = Image(
                    image_bytes=resource.data, mime_type=resource.type
                )
            elif content_type.startswith("video/"):
                video = Video(
                    video_bytes=resource.data, mime_type=resource.type
                )
            else:
                continue

        return VeoPrompt(text=prompt_text, image=image, video=video)
