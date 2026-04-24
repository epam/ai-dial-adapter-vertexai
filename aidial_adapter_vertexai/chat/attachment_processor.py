from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from logging import DEBUG
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    assert_never,
)

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import (
    MessageContentImagePart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion.request import MessageContentRefusalPart
from pydantic import BaseModel, ConfigDict

from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
    Parts,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
)
from aidial_adapter_vertexai.dial_api.resource import (
    ValidationError as ResourceValidationError,
)
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.pdf import get_pdf_page_count
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.text import decapitalize

FileTypes = dict[str, str | list[str]]

Coro = Coroutine[None, None, None]
InitValidator = Callable[[], Coro]
PostValidator = Callable[[Resource], Coro]


class AttachmentProcessor(BaseModel):
    file_types: FileTypes

    init_validator: InitValidator | None = None
    post_validator: PostValidator | None = None

    @property
    def mime_types(self) -> list[str]:
        return list(self.file_types.keys())

    @property
    def file_exts(self) -> list[str]:
        def to_list(value: str | list[str]) -> list[str]:
            return value if isinstance(value, list) else [value]

        return [
            ext for exts in self.file_types.values() for ext in to_list(exts)
        ]

    async def process(
        self, file_storage: FileStorage | None, dial_resource: DialResource
    ) -> Resource | str | None:
        try:
            type = await dial_resource.get_content_type()

            if type not in self.mime_types:
                return None

            if self.init_validator is not None:
                await self.init_validator()

            resource = await dial_resource.download(file_storage)

            if self.post_validator is not None:
                await self.post_validator(resource)

            return resource

        except Exception as e:
            log.error(
                f"Failed to process {dial_resource.entity_name}: {str(e)}"
            )
            if isinstance(e, ResourceValidationError):
                # Errors specific to a particular resource
                return e.message
            elif isinstance(e, UserError):
                # Errors not specific to any particular resource
                # typically raised by validators
                raise e
            else:
                # Unexpected runtime exceptions
                return f"Failed to process {dial_resource.entity_name}"


@dataclass(order=True, frozen=True)
class ProcessingError:
    name: str
    message: str


PartT = TypeVar("PartT")


class AttachmentProcessorsBase(BaseModel, Generic[PartT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    processors: list[AttachmentProcessor]
    file_storage: FileStorage | None
    conversation_factory: ConversationFactoryBase[PartT, Any, Any]

    errors: set[ProcessingError] = set()

    def get_error_message(self) -> str | None:
        error_list = sorted(self.errors)
        if error_list:
            msg = "The following files failed to process:\n"
            msg += "\n".join(
                f"{idx}. {error.name}: {decapitalize(error.message)}"
                for idx, error in enumerate(self.errors, start=1)
            )
            return msg

        return None

    def get_file_exts(self) -> list[str]:
        return sorted({ext for p in self.processors for ext in p.file_exts})

    async def _collect_resource(
        self, dial_resource: DialResource, resource: Resource | str
    ) -> Resource | None:
        if log.isEnabledFor(DEBUG):
            log.debug(f"resource reference: {json_dumps_short(dial_resource)}")
            log.debug(f"resource content: {json_dumps_short(resource)}")

        if isinstance(resource, str):
            name = await dial_resource.get_resource_name(self.file_storage)
            self.errors.add(ProcessingError(name=name, message=resource))
            return None
        else:
            return resource

    async def process_resource(
        self, dial_resource: DialResource
    ) -> Resource | None:
        if not self.processors:
            raise UserError("The attachments aren't supported")

        for processor in self.processors:
            resource = await processor.process(self.file_storage, dial_resource)
            if resource is not None:
                return await self._collect_resource(dial_resource, resource)

        return await self._collect_resource(
            dial_resource,
            f"The {dial_resource.entity_name} isn't one of the supported types",
        )

    async def process_message(self, message: DialMessage) -> Parts[PartT]:
        ret: Parts[PartT] = Parts()

        async def collect_resource(dial_resource: DialResource):
            resource = await self.process_resource(dial_resource)
            if resource is not None:
                part = self.conversation_factory.create_multi_modal_part(
                    resource.data, resource.type
                )
                ret.append_multi_modal_part(part, resource)

        def collect_text(text: str):
            ret.append_text_part(
                self.conversation_factory.create_text_part(text)
            )

        # Placing Images/Video parts before the text as per
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?authuser=1#image_best_practices
        for attachment in get_attachments(message):
            await collect_resource(AttachmentResource(attachment=attachment))

        content = message.content

        match content:
            case None:
                pass
            case str():
                if content:
                    collect_text(content)
            case list():
                for part in content:
                    match part:
                        case MessageContentTextPart(text=text):
                            if text:
                                collect_text(text)
                        case MessageContentImagePart(image_url=image_url):
                            await collect_resource(
                                URLResource(
                                    url=image_url.url, entity_name="image_url"
                                )
                            )
                        case MessageContentRefusalPart():
                            raise ValidationError(
                                "Refusal messages aren't supported"
                            )
                        case _:
                            assert_never(content)
            case _:
                assert_never(content)

        could_be_empty = message.role in (DialRole.SYSTEM, DialRole.DEVELOPER)
        if ret.empty() and not could_be_empty:
            collect_text(" ")

        return ret


def max_count_validator(category: str, limit: int) -> InitValidator:
    count = 0

    async def validator():
        nonlocal count
        count += 1
        if count > limit:
            raise UserError(
                f"The number of {category} files exceeds the limit ({limit})"
            )

    return validator


def max_pdf_page_count_validator(
    limit_per_request: int | None,
    limit_per_document: int | None,
) -> PostValidator:
    total_pages = 0

    async def validator(resource: Resource):
        if limit_per_document is None and limit_per_request is None:
            return

        nonlocal total_pages
        try:
            pages = await get_pdf_page_count(resource.data)
            log.debug(f"PDF page count: {pages}")
            total_pages += pages
        except Exception:
            log.exception("Failed to get PDF page count")
            raise ResourceValidationError(
                "Failed to get PDF page count"
            ) from None

        if limit_per_document is not None and pages > limit_per_document:
            raise ResourceValidationError(
                f"The number of pages in the document ({pages}) exceeds the limit ({limit_per_document})"
            )

        if limit_per_request is not None and total_pages > limit_per_request:
            raise UserError(
                f"The total number of pages in PDF documents exceeds the limit ({limit_per_request})"
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
                raise UserError(
                    f"Found documents of types {name!r} and {first!r}. "
                    "Only one type of document is supported at a time."
                )

        return validator

    return get_validator
