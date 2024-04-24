import base64
import mimetypes
from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import Attachment, Message, Role
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

        # Gemini doesn't support empty messages: neither user's nor assistant's.
        # It throws an error:
        #   400 Unable to submit request because it has an empty text parameter.
        #   Add a value to the parameter and try again.
        if content == "":
            content = " "

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
            role=from_dial_role(self.message.role),
            parts=self.to_parts(),
        )


def derive_attachment_mime_type(attachment: Attachment) -> Optional[str]:
    type = attachment.type
    url = attachment.url

    if url is not None:
        if type is None:
            # No type is provided. Trying to guess the type from the Data URL
            resource = Resource.from_data_url(url)
            if resource is not None:
                return resource.mime_type

        if type is None or "octet-stream" in type:
            # It's an arbitrary binary file. Trying to guess the type from the URL
            mime_type = mimetypes.guess_type(url)[0]
            if mime_type is not None:
                return mime_type

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


def from_dial_role(role: Role) -> str:
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