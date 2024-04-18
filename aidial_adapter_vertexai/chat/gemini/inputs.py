import base64
import mimetypes
from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import Attachment, Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.storage import FileStorage, download_file
from aidial_adapter_vertexai.utils.resource import Resource


class MessageWithInputs(BaseModel):
    message: Message
    inputs: List[Resource]

    def has_empty_content(self) -> bool:
        return (self.message.content or "").strip() == ""

    def to_content(self) -> Content:
        message = self.message
        content = message.content
        if content is None:
            raise ValidationError("Message content must be present")

        parts: List[Part] = []

        for input in self.inputs:
            data = base64.b64decode(input.base64_data, validate=True)
            parts.append(Part.from_data(data=data, mime_type=input.mime_type))

        # Placing Images/Video before the text as per
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?authuser=1#image_best_practices
        parts.append(Part.from_text(content))

        return Content(role=get_part_role(message.role), parts=parts)


def derive_attachment_mime_type(attachment: Attachment) -> Optional[str]:
    type = attachment.type

    if type is None:
        # No type is provided. Trying to guess the type from the Data URL.
        if attachment.url is not None:
            resource = Resource.from_data_url(attachment.url)
            if resource is not None:
                return resource.mime_type
        return None

    if "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        url = attachment.url
        if url is not None:
            url_type = mimetypes.guess_type(url)[0]
            if url_type is not None:
                return url_type
        return None

    return type


async def download_attachment(
    file_storage: Optional[FileStorage], attachment: Attachment
) -> bytes:
    if attachment.data is not None:
        return attachment.data.encode()

    if attachment.url is not None:
        attachment_link: str = attachment.url

        resource = Resource.from_data_url(attachment_link)
        if resource is not None:
            return resource.data

        if file_storage is not None:
            url = file_storage.attachment_link_to_url(attachment_link)
            return await file_storage.download_file(url)

        return await download_file(attachment_link)

    raise ValueError("Invalid attachment: neither url nor data is provided")


def get_part_role(role: Role) -> str:
    match role:
        case Role.SYSTEM:
            raise ValidationError(
                "System messages other than the first system message are not allowed"
            )
        case Role.USER:
            return ChatSession._USER_ROLE
        case Role.ASSISTANT:
            return ChatSession._MODEL_ROLE
        case Role.FUNCTION:
            raise ValidationError("Function messages are not supported")
        case Role.TOOL:
            raise ValidationError("Tool messages are not supported")
        case _:
            assert_never(role)
