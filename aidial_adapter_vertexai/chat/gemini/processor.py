import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import DEBUG
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    Set,
    Union,
    assert_never,
)

from aidial_sdk.chat_completion import (
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion.request import Role
from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from pydantic.v1 import BaseModel, Field
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.prompt.base import (
    ContentT,
    GeminiConversation,
    GeminiConversationT,
    GeminiGenAIConversation,
    PartT,
)
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
        self, file_storage: FileStorage | None, dial_resource: DialResource
    ) -> Optional[Resource | str]:
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
                f"Failed to download {dial_resource.entity_name}: {str(e)}"
            )
            if isinstance(e, ResourceValidationError):
                return e.message
            return f"Failed to download {dial_resource.entity_name}"


@dataclass(order=True, frozen=True)
class ProcessingError:
    name: str
    message: str


class ConversationFactoryBase(
    ABC, Generic[PartT, ContentT, GeminiConversationT]
):
    @abstractmethod
    def create_multi_modal_part(self, data: bytes, mime_type: str) -> PartT: ...

    @abstractmethod
    def create_text_part(self, text: str) -> PartT: ...

    @abstractmethod
    def create_function_call_part(self, name: str, args: str) -> PartT: ...

    @abstractmethod
    def create_function_result_part(self, name: str, args: str) -> PartT: ...

    @abstractmethod
    def create_content(self, role: Role, parts: List[PartT]) -> ContentT: ...

    @abstractmethod
    def create_conversation(
        self, system_instruction: List[PartT] | None, contents: List[ContentT]
    ) -> GeminiConversationT: ...


class ConversationFactory(
    ConversationFactoryBase[Part, Content, GeminiConversation]
):
    @staticmethod
    def _to_gemini_role(role: Role) -> str:
        match role:
            case Role.SYSTEM:
                raise ValidationError(
                    "System messages other than the first system message are not allowed"
                )
            case Role.USER | Role.FUNCTION | Role.TOOL:
                return ChatSession._USER_ROLE
            case Role.ASSISTANT:
                return ChatSession._MODEL_ROLE
            case _:
                assert_never(role)

    def create_multi_modal_part(self, data: bytes, mime_type: str) -> Part:
        return Part.from_data(data=data, mime_type=mime_type)

    def create_text_part(self, text: str) -> Part:
        return Part.from_text(text)

    def create_function_call_part(self, name: str, args: str) -> Part:
        try:
            args = json.loads(args)
            return Part.from_dict(
                {"function_call": {"name": name, "args": args}}
            )
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            )

    def create_function_result_part(self, name: str, args: str) -> Part:
        try:
            args = json.loads(args)
        except Exception:
            args = args

        if isinstance(args, dict):
            return Part.from_function_response(name, args)

        return Part.from_function_response(name, {"content": args})

    def create_content(self, role: Role, parts: List[Part]) -> Content:
        return Content(role=self._to_gemini_role(role), parts=parts)

    def create_conversation(
        self, system_instruction: List[Part] | None, contents: List[Content]
    ) -> GeminiConversation:
        return GeminiConversation(
            system_instruction=system_instruction, contents=contents
        )


class GenAIConversationFactory(
    ConversationFactoryBase[GenAIPart, GenAIContent, GeminiGenAIConversation]
):
    @staticmethod
    def to_gemini_genai_role(role: Role) -> str:
        match role:
            case Role.SYSTEM:
                raise ValidationError(
                    "System messages other than the first system message are not allowed"
                )
            case Role.USER | Role.FUNCTION | Role.TOOL:
                return "user"
            case Role.ASSISTANT:
                return "model"
            case _:
                assert_never(role)

    def create_multi_modal_part(self, data: bytes, mime_type: str) -> GenAIPart:
        return GenAIPart.from_bytes(data=data, mime_type=mime_type)

    def create_text_part(self, text: str) -> GenAIPart:
        return GenAIPart.from_text(text)

    def create_function_call_part(self, name: str, args: str) -> GenAIPart:
        try:
            return GenAIPart.from_function_call(name, json.loads(args))
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            )

    def create_function_result_part(self, name: str, args: str) -> GenAIPart:
        try:
            processed_args = json.loads(args)
        except Exception:
            processed_args = args

        if isinstance(processed_args, dict):
            return GenAIPart.from_function_response(name, processed_args)

        return GenAIPart.from_function_response(
            name, {"output": processed_args}
        )

    def create_content(
        self, role: Role, parts: List[GenAIPart]
    ) -> GenAIContent:
        return GenAIContent(role=self.to_gemini_genai_role(role), parts=parts)

    def create_conversation(
        self,
        system_instruction: List[GenAIPart] | None,
        contents: List[GenAIContent],
    ) -> GeminiGenAIConversation:
        return GeminiGenAIConversation(
            system_instruction=system_instruction, contents=contents
        )


class AttachmentProcessorsBase(BaseModel, ABC, Generic[PartT]):
    class Config:
        arbitrary_types_allowed = True  # for errors

    processors: List[AttachmentProcessor]
    file_storage: FileStorage | None
    conversation_factory: ConversationFactoryBase[PartT, Any, Any]

    errors: Set[ProcessingError] = Field(default_factory=set)
    resource_count: int = 0

    def get_error_message(self) -> str | None:
        error_list = sorted(list(self.errors))
        if error_list:
            msg = "The following files failed to process:\n"
            msg += "\n".join(
                f"{idx}. {error.name}: {decapitalize(error.message)}"
                for idx, error in enumerate(self.errors, start=1)
            )
            return msg

        return None

    def get_file_exts(self) -> List[str]:
        return sorted({ext for p in self.processors for ext in p.file_exts})

    def get_mime_types(self) -> List[str]:
        return sorted({ty for p in self.processors for ty in p.mime_types})

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
            self.resource_count += 1
            return resource

    async def process_resource(
        self, dial_resource: DialResource
    ) -> Resource | None:
        if not self.processors:
            raise ValidationError("The attachments aren't supported")

        for processor in self.processors:
            resource = await processor.process(self.file_storage, dial_resource)
            if resource is not None:
                return await self._collect_resource(dial_resource, resource)

        return await self._collect_resource(
            dial_resource,
            f"The {dial_resource.entity_name} isn't one of the supported types",
        )

    async def process_message(self, message: Message) -> List[PartT]:
        ret: List[PartT] = []

        async def collect_resource(dial_resource: DialResource):
            resource = await self.process_resource(dial_resource)
            if resource is not None:
                ret.append(
                    self.conversation_factory.create_multi_modal_part(
                        resource.data, resource.type
                    )
                )

        def collect_text(text: str):
            ret.append(self.conversation_factory.create_text_part(text))

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
                            collect_text(text)
                        case MessageContentImagePart(image_url=image_url):
                            await collect_resource(
                                URLResource(
                                    url=image_url.url, entity_name="image_url"
                                )
                            )
            case _:
                assert_never(content)

        return ret


class AttachmentProcessors(AttachmentProcessorsBase[Part]):
    pass


class AttachmentProcessorsGenAI(AttachmentProcessorsBase[GenAIPart]):
    pass


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
