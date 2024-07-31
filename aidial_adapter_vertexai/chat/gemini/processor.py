from logging import DEBUG
from typing import (
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Union,
)

from aidial_sdk.chat_completion import Attachment, Message
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithResources
from aidial_adapter_vertexai.dial_api.attachments import (
    derive_attachment_mime_type,
    download_attachment,
)
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.pdf import get_pdf_page_count
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.text import format_ordinal

FileTypes = Dict[str, Union[str, List[str]]]

Coro = Coroutine[None, None, None]
InitValidator = Callable[[], Coro]
PostValidator = Callable[[Resource], Coro]


class AttachmentProcessor(BaseModel):
    file_types: FileTypes

    init_validator: InitValidator | None = None
    post_validator: PostValidator | None = None

    @property
    def mime_types(self) -> List[str]:
        return list(self.file_types.keys())

    @property
    def file_exts(self) -> List[str]:
        def to_list(value: Union[str, List[str]]) -> List[str]:
            return value if isinstance(value, list) else [value]

        return [
            ext for exts in self.file_types.values() for ext in to_list(exts)
        ]

    async def process(
        self, file_storage: Optional[FileStorage], attachment: Attachment
    ) -> Optional[Resource | str]:
        try:
            mime_type = derive_attachment_mime_type(attachment)
            if mime_type is None:
                return "Can't derive media type of the attachment"

            if mime_type not in self.mime_types:
                return None

            if self.init_validator is not None:
                await self.init_validator()

            data = await download_attachment(file_storage, attachment)
            resource = Resource(mime_type=mime_type, data=data)

            if self.post_validator is not None:
                await self.post_validator(resource)

            return resource

        except ValidationError as e:
            log.error(f"Validation error: {e.message}")
            return e.message

        except Exception as e:
            log.error(f"Failed to download file: {str(e)}")
            return "Failed to download file"


async def process_attachment(
    processors: List[AttachmentProcessor],
    file_storage: Optional[FileStorage],
    attachment: Attachment,
) -> Resource | str:
    for processor in processors:
        resource = await processor.process(file_storage, attachment)
        if resource is not None:
            return resource

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
    processors: List[AttachmentProcessor],
    file_storage: Optional[FileStorage],
    attachments: List[Attachment],
) -> List[Resource] | ProcessingErrors:
    if len(attachments) == 0:
        return []

    if log.isEnabledFor(DEBUG):
        log.debug(f"attachments: {json_dumps_short(attachments)}")

    download_results: List[Resource | str] = [
        await process_attachment(processors, file_storage, attachment)
        for attachment in attachments
    ]

    if log.isEnabledFor(DEBUG):
        log.debug(f"processing results: {json_dumps_short(download_results)}")

    ret: List[Resource] = []
    errors: List[Tuple[int, str]] = []

    for idx, result in enumerate(download_results):
        if isinstance(result, Resource):
            ret.append(result)
        else:
            errors.append((idx, result))

    if len(errors) > 0:
        log.error(f"processing errors: {errors}")
        return ProcessingErrors(errors=errors)

    return ret


async def process_message(
    processors: List[AttachmentProcessor],
    file_storage: Optional[FileStorage],
    message: Message,
) -> MessageWithResources | ProcessingErrors:

    attachments = get_attachments(message)
    resources = await process_attachments(processors, file_storage, attachments)

    if isinstance(resources, ProcessingErrors):
        return resources

    return MessageWithResources(message=message, resources=resources)


async def process_messages(
    processors: List[AttachmentProcessor],
    file_storage: Optional[FileStorage],
    messages: List[Message],
) -> List[MessageWithResources] | str:
    ret: List[MessageWithResources] = []
    errors: Dict[int, ProcessingErrors] = {}

    for idx, message in enumerate(messages):
        result = await process_message(processors, file_storage, message)
        if isinstance(result, ProcessingErrors):
            errors[idx] = result
        else:
            ret.append(result)

    if errors:
        return ProcessingErrors.format_error_message(errors, len(messages))

    return ret


def max_count_validator(limit: int) -> InitValidator:
    count = 0

    async def validator():
        nonlocal count
        count += 1
        if count > limit:
            raise ValidationError(
                f"The number of files exceeds the limit ({limit})"
            )

    return validator


def max_pdf_page_count_validator(limit: int) -> PostValidator:
    count = 0

    async def validator(resource: Resource):
        nonlocal count
        try:
            pages = await get_pdf_page_count(resource.data)
            log.debug(f"PDF page count: {pages}")
            count += pages
        except Exception:
            log.exception("Failed to get PDF page count")
            raise ValidationError("Failed to get PDF page count")

        if count > limit:
            raise ValidationError(
                f"The total number of PDF pages exceeds the limit ({limit})"
            )

    return validator


P = ParamSpec("P")


def seq_validators(*validators: Callable[P, Coro] | None) -> Callable[P, Coro]:
    async def validator(*args: P.args, **kwargs: P.kwargs) -> None:
        for v in validators:
            if v is not None:
                await v(*args, **kwargs)

    return validator


def exclusive_validator() -> Callable[[str], InitValidator]:
    first: str | None = None

    def get_validator(name: str) -> InitValidator:
        async def validator():
            nonlocal first
            if first is None:
                first = name
            elif first != name:
                raise ValidationError(
                    f"The document type is {name!r}. "
                    f"However, one of the documents processed earlier was of {first!r} type. "
                    "Only one type of document is supported at a time."
                )

        return validator

    return get_validator
