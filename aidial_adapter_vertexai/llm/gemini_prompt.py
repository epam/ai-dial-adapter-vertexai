import base64
from typing import Dict, List, Optional, Tuple, assert_never, cast

from aidial_sdk.chat_completion import Attachment, Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.llm.download_image import download_image
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.utils.image_data_url import ImageDataURL
from aidial_adapter_vertexai.utils.log_config import app_logger as logger
from aidial_adapter_vertexai.utils.storage import FileStorage
from aidial_adapter_vertexai.utils.text import format_ordinal


class GeminiPrompt(BaseModel):
    history: List[Content]
    prompt: List[Part]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    async def parse(
        cls,
        file_storage: Optional[FileStorage],
        download_images: bool,
        messages: List[Message],
    ) -> "GeminiPrompt":
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        res = await transform_messages(file_storage, download_images, messages)

        if isinstance(res, str):
            raise ValidationError(res)

        history, _ = res

        return cls(history=history[:-1], prompt=history[-1].parts)

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]


def get_part_role(role: Role) -> str:
    match role:
        case Role.SYSTEM:
            raise ValidationError(
                "System messages are not allowed in Gemini models"
            )
        case Role.USER:
            return ChatSession._USER_ROLE
        case Role.ASSISTANT:
            return ChatSession._MODEL_ROLE
        case Role.FUNCTION:
            raise ValidationError("Function messages are not supported")
        case _:
            assert_never(role)


def get_attachments(message: Message) -> List[Attachment]:
    custom_content = message.custom_content
    if custom_content is None:
        return []
    return custom_content.attachments or []


class DownloadErrors(BaseModel):
    errors: List[Tuple[int, str]]


async def download_image_attachments(
    file_storage: Optional[FileStorage], attachments: List[Attachment]
) -> List[ImageDataURL] | DownloadErrors:
    logger.debug(f"original attachments: {attachments}")

    download_results: List[ImageDataURL | str] = [
        await download_image(file_storage, attachment)
        for attachment in attachments
    ]

    logger.debug(f"download results: {download_results}")

    errors: List[Tuple[int, str]] = [
        (idx, result)
        for idx, result in enumerate(download_results)
        if isinstance(result, str)
    ]

    if len(errors) > 0:
        logger.debug(f"download errors: {errors}")
        return DownloadErrors(errors=errors)

    return cast(List[ImageDataURL], download_results)


async def transform_message(
    file_storage: Optional[FileStorage], download_images: bool, message: Message
) -> Tuple[Content, int] | DownloadErrors:
    content = message.content
    if content is None:
        raise ValueError("Message content must be present")

    attachments = get_attachments(message)

    parts: List[Part] = [Part.from_text(content)]

    if download_images:
        images = await download_image_attachments(file_storage, attachments)
        if isinstance(images, DownloadErrors):
            return images

        for image in images:
            data = base64.b64decode(image.data)
            parts.append(Part.from_data(data=data, mime_type=image.type))

    new_message = Content(role=get_part_role(message.role), parts=parts)
    return new_message, len(parts) - 1


def format_error_message(errors: Dict[int, DownloadErrors]) -> str:
    msg = "Some of the image attachments failed to download:"
    for i, error in errors.items():
        msg += f"\n- {format_ordinal(i)} message from end:"
        for j, err in error.errors:
            msg += f"\n  - {format_ordinal(j + 1)} attachment: {err}"
    return msg


async def transform_messages(
    file_storage: Optional[FileStorage],
    download_images: bool,
    messages: List[Message],
) -> Tuple[List[Content], int] | str:
    new_messages: List[Content] = []
    image_stats = 0

    errors: Dict[int, DownloadErrors] = {}

    n = len(messages)
    for idx, message in enumerate(messages):
        result = await transform_message(file_storage, download_images, message)
        if isinstance(result, DownloadErrors):
            errors[n - idx] = result
        else:
            new_message, stats = result
            new_messages.append(new_message)
            image_stats += stats

    if errors:
        return format_error_message(errors)

    return new_messages, image_stats
