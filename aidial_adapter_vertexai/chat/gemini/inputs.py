import base64
import mimetypes
from logging import DEBUG
from typing import Dict, List, Optional, Tuple, assert_never

from aidial_sdk.chat_completion import Attachment, Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_vertexai.utils.data_url import DataURL
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.text import format_ordinal


class MessageWithInputs(BaseModel):
    message: Message
    inputs: List[DataURL]

    def has_empty_content(self) -> bool:
        return (self.message.content or "").strip() == ""

    def to_content(self) -> Content:
        message = self.message
        content = message.content
        if content is None:
            raise ValidationError("Message content must be present")

        parts: List[Part] = []

        for input in self.inputs:
            data = base64.b64decode(input.data, validate=True)
            parts.append(Part.from_data(data=data, mime_type=input.type))

        # Placing Images/Video before the text as per
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?authuser=1#image_best_practices
        parts.append(Part.from_text(content))

        return Content(role=get_part_role(message.role), parts=parts)


class DownloadErrors(BaseModel):
    """Download errors for a particular message"""

    errors: List[Tuple[int, str]]
    """List of pairs (attachment index, error message)"""


def guess_attachment_type(attachment: Attachment) -> Optional[str]:
    type = attachment.type
    if type is None:
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


async def download_file(
    file_storage: Optional[FileStorage],
    mime_types: List[str],
    attachment: Attachment,
) -> DataURL | str:
    try:
        type = guess_attachment_type(attachment)
        if type is None:
            return "Can't derive media type of the attachment"
        elif type not in mime_types:
            return f"The attachment isn't one of the supported types: {type}"

        if attachment.data is not None:
            return DataURL(type=type, data=attachment.data)

        if attachment.url is not None:
            attachment_link: str = attachment.url

            data_url = DataURL.from_data_url(attachment_link)
            if data_url is not None:
                if data_url.type not in mime_types:
                    return f"The attachment data isn't one of the supported types: {data_url.type}"
                return data_url

            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return DataURL(type=type, data=data)

        return "Invalid attachment"

    except Exception as e:
        log.debug(f"Failed to download file: {e}")
        return "Failed to download file"


async def download_files(
    file_storage: Optional[FileStorage],
    file_types: List[str],
    attachments: List[Attachment],
) -> List[DataURL] | DownloadErrors:
    if log.isEnabledFor(DEBUG):
        log.debug(f"original attachments: {json_dumps_short(attachments)}")

    download_results: List[DataURL | str] = [
        await download_file(file_storage, file_types, attachment)
        for attachment in attachments
    ]

    if log.isEnabledFor(DEBUG):
        log.debug(f"download results: {json_dumps_short(download_results)}")

    ret: List[DataURL] = []
    errors: List[Tuple[int, str]] = []

    for idx, result in enumerate(download_results):
        if isinstance(result, DataURL):
            ret.append(result)
        else:
            errors.append((idx, result))

    if len(errors) > 0:
        log.debug(f"download errors: {errors}")
        return DownloadErrors(errors=errors)

    return ret


async def download_inputs_from_message(
    file_storage: Optional[FileStorage],
    file_types: Optional[List[str]],
    message: Message,
) -> MessageWithInputs | DownloadErrors:
    inputs: List[DataURL] = []
    if file_types is not None:
        attachments = get_attachments(message)
        res = await download_files(file_storage, file_types, attachments)
        if isinstance(res, DownloadErrors):
            return res
        inputs = res

    return MessageWithInputs(message=message, inputs=inputs)


async def download_inputs(
    file_storage: Optional[FileStorage],
    file_types: Optional[List[str]],
    messages: List[Message],
) -> List[MessageWithInputs] | str:
    ret: List[MessageWithInputs] = []
    errors: Dict[int, DownloadErrors] = {}

    for idx, message in enumerate(messages):
        result = await download_inputs_from_message(
            file_storage, file_types, message
        )
        if isinstance(result, DownloadErrors):
            errors[idx] = result
        else:
            ret.append(result)

    if errors:
        return format_error_message(errors, len(messages))

    return ret


def format_error_message(errors: Dict[int, DownloadErrors], n: int) -> str:
    msg = "Some of the attachments has failed to download:"
    for i, error in errors.items():
        msg += f"\n- {format_ordinal(n - i)} message from end:"
        for j, err in error.errors:
            msg += f"\n  - {format_ordinal(j + 1)} attachment: {err}"
    return msg


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
