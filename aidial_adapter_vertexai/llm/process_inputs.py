import mimetypes
from logging import DEBUG
from typing import Dict, List, Optional, Tuple

from aidial_sdk.chat_completion import Attachment, Message
from pydantic import BaseModel

from aidial_adapter_vertexai.universal_api.request import get_attachments
from aidial_adapter_vertexai.universal_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_vertexai.utils.image_data_url import ImageDataURL
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.text import format_ordinal


class MessageWithInputs(BaseModel):
    message: Message
    image_inputs: List[ImageDataURL]


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


async def download_image(
    file_storage: Optional[FileStorage],
    image_types: List[str],
    attachment: Attachment,
) -> ImageDataURL | str:
    try:
        type = guess_attachment_type(attachment)
        if type is None:
            return "Can't derive media type of the attachment"
        elif type not in image_types:
            return f"The attachment isn't one of the supported types: {type}"

        if attachment.data is not None:
            return ImageDataURL(type=type, data=attachment.data)

        if attachment.url is not None:
            attachment_link: str = attachment.url

            image_url = ImageDataURL.from_data_url(attachment_link)
            if image_url is not None:
                if image_url.type in image_types:
                    return image_url
                else:
                    return (
                        "The image attachment isn't one of the supported types"
                    )

            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return ImageDataURL(type=type, data=data)

        return "Invalid attachment"

    except Exception as e:
        log.debug(f"Failed to download image: {e}")
        return "Failed to download image"


async def download_images(
    file_storage: Optional[FileStorage],
    image_types: List[str],
    attachments: List[Attachment],
) -> List[ImageDataURL] | DownloadErrors:
    if log.isEnabledFor(DEBUG):
        log.debug(f"original attachments: {json_dumps_short(attachments)}")

    download_results: List[ImageDataURL | str] = [
        await download_image(file_storage, image_types, attachment)
        for attachment in attachments
    ]

    if log.isEnabledFor(DEBUG):
        log.debug(f"download results: {json_dumps_short(download_results)}")

    ret: List[ImageDataURL] = []
    errors: List[Tuple[int, str]] = []

    for idx, result in enumerate(download_results):
        if isinstance(result, ImageDataURL):
            ret.append(result)
        else:
            errors.append((idx, result))

    if len(errors) > 0:
        log.debug(f"download errors: {errors}")
        return DownloadErrors(errors=errors)
    else:
        return ret


async def download_inputs_from_message(
    file_storage: Optional[FileStorage],
    image_types: Optional[List[str]],
    message: Message,
) -> MessageWithInputs | DownloadErrors:
    inputs: List[ImageDataURL] = []
    if image_types is not None:
        attachments = get_attachments(message)
        res = await download_images(file_storage, image_types, attachments)
        if isinstance(res, DownloadErrors):
            return res
        inputs = res

    return MessageWithInputs(message=message, image_inputs=inputs)


async def download_inputs(
    file_storage: Optional[FileStorage],
    image_types: Optional[List[str]],
    messages: List[Message],
) -> List[MessageWithInputs] | str:
    ret: List[MessageWithInputs] = []
    errors: Dict[int, DownloadErrors] = {}

    for idx, message in enumerate(messages):
        result = await download_inputs_from_message(
            file_storage, image_types, message
        )
        if isinstance(result, DownloadErrors):
            errors[idx] = result
        else:
            ret.append(result)

    if errors:
        return format_error_message(errors, len(messages))

    return ret


def format_error_message(errors: Dict[int, DownloadErrors], n: int) -> str:
    msg = "Some of the image attachments failed to download:"
    for i, error in errors.items():
        msg += f"\n- {format_ordinal(n - i)} message from end:"
        for j, err in error.errors:
            msg += f"\n  - {format_ordinal(j + 1)} attachment: {err}"
    return msg
