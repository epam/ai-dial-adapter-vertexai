import base64
import mimetypes
from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import Attachment, CustomContent, Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.storage import FileStorage, download_file
from aidial_adapter_vertexai.utils.resource import Resource


class MessageWithResources(BaseModel):
    message: Message
    resources: List[Resource]

    @property
    def content(self) -> str:
        content = self.message.content
        if content is None:
            raise ValidationError("Message content must be present")

        if content.strip() == "":
            raise ValidationError("Message with empty content isn't allowed")

        return content

    def to_text(self) -> str:
        if len(self.resources) > 0:
            raise ValidationError("Inputs are not supported for text messages")

        return self.content

    def to_parts(self) -> List[Part]:
        parts = [resource.to_part() for resource in self.resources]

        # Placing Images/Video before the text as per
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?authuser=1#image_best_practices
        parts.append(Part.from_text(self.content))

        return parts

    def to_content(self) -> Content:
        return Content(
            role=get_part_role(self.message.role),
            parts=self.to_parts(),
        )

    def to_message(self) -> Message:
        attachments = [input.to_attachment() for input in self.resources]
        return Message(
            role=self.message.role,
            content=self.message.content,
            custom_content=CustomContent(attachments=attachments),
        )


def derive_attachment_mime_type(attachment: Attachment) -> Optional[str]:
    type = attachment.type

    if type is None:
        # No type is provided. Trying to guess the type from the Data URL
        if attachment.url is not None:
            resource = Resource.from_data_url(attachment.url)
            if resource is not None:
                return resource.mime_type
        return None

    if "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL
        url = attachment.url
        if url is not None:
            mime_type = mimetypes.guess_type(url)[0]
            if mime_type is not None:
                return mime_type
        return None

    return type


async def download_attachment(
    file_storage: Optional[FileStorage], attachment: Attachment
) -> bytes:
    if attachment.data is not None:
        return base64.b64decode(attachment.data, validate=True)

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
