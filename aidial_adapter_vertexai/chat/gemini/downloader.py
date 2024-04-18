from logging import DEBUG
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from aidial_sdk.chat_completion import Attachment, Message
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.gemini.inputs import (
    MessageWithInputs,
    derive_attachment_mime_type,
    download_attachment,
)
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.data_url import DataURL
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.pdf import get_pdf_page_count
from aidial_adapter_vertexai.utils.text import format_ordinal

FileTypes = Dict[str, Union[str, List[str]]]

AnyCoro = Coroutine[None, None, Any]


class Downloader(BaseModel):
    file_types: FileTypes

    file_pre_validator: Callable[[], AnyCoro] | None = None
    file_post_validator: Callable[[DataURL], AnyCoro] | None = None

    @property
    def mime_types(self) -> List[str]:
        return list(self.file_types.keys())

    @property
    def file_exts(self) -> List[str]:
        def flatten(value: Union[str, List[str]]):
            return value if isinstance(value, list) else [value]

        return [
            ext for exts in self.file_types.values() for ext in flatten(exts)
        ]

    async def process_attachment(
        self, file_storage: Optional[FileStorage], attachment: Attachment
    ) -> Optional[DataURL | str]:
        try:
            mime_type = derive_attachment_mime_type(attachment)
            if mime_type is None:
                return "Can't derive media type of the attachment"

            if mime_type not in self.mime_types:
                return None

            if self.file_pre_validator is not None:
                await self.file_pre_validator()

            data = await download_attachment(file_storage, attachment)
            data_url = DataURL(mime_type=mime_type, data=data)

            if self.file_post_validator is not None:
                await self.file_post_validator(data_url)

            return data_url

        except Exception as e:
            log.error(f"Failed to download file: {str(e)}")
            return "Failed to download file"


async def process_attachment(
    downloaders: List[Downloader],
    file_storage: Optional[FileStorage],
    attachment: Attachment,
) -> DataURL | str:
    for downloader in downloaders:
        data_url = await downloader.process_attachment(file_storage, attachment)
        if data_url is not None:
            return data_url

    return "The attachment isn't one of the supported types"


class ProcessingErrors(BaseModel):
    """Processing errors for a particular message"""

    errors: List[Tuple[int, str]]
    """List of pairs (attachment index, error message)"""

    @staticmethod
    def format_error_message(
        errors: Dict[int, "ProcessingErrors"], n: int
    ) -> str:
        msg = "Some of the attachments failed to process:"
        for i, error in errors.items():
            msg += f"\n- {format_ordinal(n - i)} message from end:"
            for j, err in error.errors:
                msg += f"\n  - {format_ordinal(j + 1)} attachment: {err}"
        return msg


async def process_attachments(
    downloaders: List[Downloader],
    file_storage: Optional[FileStorage],
    attachments: List[Attachment],
) -> List[DataURL] | ProcessingErrors:
    if log.isEnabledFor(DEBUG):
        log.debug(f"original attachments: {json_dumps_short(attachments)}")

    download_results: List[DataURL | str] = [
        await process_attachment(downloaders, file_storage, attachment)
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
        log.error(f"download errors: {errors}")
        return ProcessingErrors(errors=errors)

    return ret


async def process_message(
    downloaders: List[Downloader],
    file_storage: Optional[FileStorage],
    message: Message,
) -> MessageWithInputs | ProcessingErrors:

    attachments = get_attachments(message)
    inputs = await process_attachments(downloaders, file_storage, attachments)

    if isinstance(inputs, ProcessingErrors):
        return inputs

    return MessageWithInputs(message=message, inputs=inputs)


async def process_messages(
    downloaders: List[Downloader],
    file_storage: Optional[FileStorage],
    messages: List[Message],
) -> List[MessageWithInputs] | str:
    ret: List[MessageWithInputs] = []
    errors: Dict[int, ProcessingErrors] = {}

    for idx, message in enumerate(messages):
        result = await process_message(downloaders, file_storage, message)
        if isinstance(result, ProcessingErrors):
            errors[idx] = result
        else:
            ret.append(result)

    if errors:
        return ProcessingErrors.format_error_message(errors, len(messages))

    return ret


def max_count(limit: int) -> Callable[[], AnyCoro]:
    count = 0

    async def validator():
        nonlocal count
        count += 1
        if count > limit:
            raise ValueError(f"The number of files exceeds the limit ({limit})")

    return validator


def max_pdf_page_count(limit: int) -> Callable[[DataURL], AnyCoro]:
    count = 0

    async def validator(data_url: DataURL):
        nonlocal count
        try:
            count += await get_pdf_page_count(data_url.data)
        except Exception as e:
            log.debug(f"Failed to get PDF page count: {e}")
            raise ValueError("Failed to get PDF page count")

        if count > limit:
            raise ValueError(
                f"The total number of PDF pages exceeds the limit ({limit})"
            )

    return validator
