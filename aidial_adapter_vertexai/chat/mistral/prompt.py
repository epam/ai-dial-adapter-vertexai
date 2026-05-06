import re
from dataclasses import dataclass
from typing import Any, TypeVar, assert_never

from aidial_sdk.chat_completion import (
    Message,
    MessageContentAudioPart,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    Role,
)
from aidial_sdk.chat_completion.request import (
    MessageContentRefusalPart,
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatText,
)
from aidial_sdk.chat_completion.request import ToolChoice as DialToolChoice
from aidial_sdk.exceptions import RequestValidationError
from mistralai.client.models import (
    AssistantMessage,
    ContentChunk,
    Function,
    FunctionCall,
    FunctionName,
    ImageURL,
    ImageURLChunk,
    JSONSchema,
    ResponseFormat,
    SystemMessage,
    SystemMessageContentChunks,
    TextChunk,
    Tool,
    ToolCall,
    ToolChoice,
    ToolChoiceEnum,
    ToolMessage,
    UserMessage,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import (
    ModelParameters,
    get_attachments,
)
from aidial_adapter_vertexai.dial_api.resource import AttachmentResource
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.json import (
    inline_local_json_refs,
    to_json_object_or_string,
)

_EMPTY_OBJECT_JSON_SCHEMA = {"type": "object", "properties": {}}
_MISTRAL_TOOL_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{9}$")
_DEPRECATED_FUNCTION_API = "The deployment doesn't support the deprecated API for functions. Please use tools instead."
_T = TypeVar("_T")

PromptMessage = SystemMessage | AssistantMessage | UserMessage | ToolMessage


@dataclass
class MistralPrompt:
    messages: list[PromptMessage]
    tools: list[Tool] | None = None
    use_tool_api: bool = True
    tool_choice: ToolChoice | ToolChoiceEnum | None = None
    response_format: ResponseFormat | None = None

    # NOTE: These properties intentionally expose untyped wire data.
    # They are passed directly to SDK clients to avoid type conflicts between
    # `mistralai.client` and `mistralai.gcp.client` models that are runtime-compatible.
    @property
    def messages_unwrap(self) -> Any:
        return [m.model_dump() for m in self.messages]

    @property
    def tools_unwrap(self) -> Any | None:
        if self.tools is None:
            return None
        return [m.model_dump() for m in self.tools]

    @property
    def tool_choice_unwrap(self) -> Any | None:
        if self.tool_choice is None:
            return None
        if isinstance(self.tool_choice, str):
            return self.tool_choice
        return self.tool_choice.model_dump()

    @property
    def response_format_unwrap(self) -> Any | None:
        if self.response_format is None:
            return None
        return self.response_format.model_dump()


class MistralPromptParser:
    @classmethod
    async def parse(
        cls,
        params: ModelParameters,
        tools: ToolsConfig,
        file_storage: FileStorage | None,
        messages: list[Message],
    ) -> MistralPrompt:
        if len(messages) == 0:
            raise ValidationError("The list of messages must not be empty")

        return MistralPrompt(
            messages=await cls._to_mistral_messages(messages, file_storage),
            tools=cls._to_mistral_tools(tools),
            use_tool_api=tools.is_tool,
            tool_choice=cls._to_mistral_tool_choice(tools),
            response_format=cls._to_mistral_response_format(params),
        )

    @staticmethod
    def _to_mistral_response_format(
        params: ModelParameters,
    ) -> ResponseFormat | None:
        format_ = params.response_format
        match format_:
            case None:
                return None
            case ResponseFormatText():
                return ResponseFormat(type="text")
            case ResponseFormatJsonObject():
                return ResponseFormat(type="json_object")
            case ResponseFormatJsonSchema():
                schema = format_.json_schema
                return ResponseFormat(
                    type="json_schema",
                    json_schema=JSONSchema(
                        name=schema.name,
                        description=schema.description,
                        schema_definition=schema.schema_,
                        strict=schema.strict,
                    ),
                )
            case _:
                assert_never(format_)

    @staticmethod
    def _to_mistral_tools(tools: ToolsConfig) -> list[Tool] | None:
        if tools.is_empty():
            return None

        return [
            Tool(
                function=Function(
                    name=tool.function.name,
                    description=tool.function.description,
                    strict=tool.function.strict,
                    parameters=_normalize_tool_parameters(
                        tool.function.parameters
                    )
                    or _EMPTY_OBJECT_JSON_SCHEMA,
                )
            )
            for tool in tools.tools
        ]

    @staticmethod
    def _to_mistral_tool_choice(
        tools: ToolsConfig,
    ) -> ToolChoice | ToolChoiceEnum | None:
        if tools.is_empty():
            return None

        choice = tools.tool_choice
        match choice:
            case str():
                return choice
            case DialToolChoice():
                return ToolChoice(
                    function=FunctionName(name=choice.function.name)
                )
            case _:
                assert_never(choice)

    @classmethod
    async def _to_mistral_messages(
        cls,
        messages: list[Message],
        file_storage: FileStorage | None,
    ) -> list[PromptMessage]:
        prompt: list[PromptMessage] = []
        function_call_idx = 0
        pending_legacy_function_call_ids: list[str] = []
        tool_call_id_map: dict[str, str] = {}

        for message in messages:
            match message.role:
                case Role.SYSTEM | Role.DEVELOPER:
                    content = await cls._merge_content_with_attachments(
                        message,
                        file_storage=file_storage,
                    )
                    if isinstance(content, str):
                        prompt.append(SystemMessage(content=content))
                        continue

                    system_chunks: list[SystemMessageContentChunks] = []
                    for chunk in content:
                        if isinstance(chunk, TextChunk):
                            system_chunks.append(chunk)
                            continue
                        raise ValidationError(
                            "System and developer messages support only text content"
                        )

                    prompt.append(SystemMessage(content=system_chunks))
                case Role.USER:
                    prompt.append(
                        UserMessage(
                            content=await cls._merge_content_with_attachments(
                                message,
                                file_storage=file_storage,
                            )
                        )
                    )
                case Role.ASSISTANT:
                    if message.function_call is not None:
                        raise RequestValidationError(_DEPRECATED_FUNCTION_API)

                    content = await cls._merge_content_with_attachments(
                        message,
                        file_storage=file_storage,
                    )

                    # Some upstreams reject assistant messages when content is effectively empty.
                    assistant_msg = AssistantMessage(content=content or " ")

                    if message.tool_calls:
                        assistant_msg.tool_calls = [
                            ToolCall(
                                id=to_mistral_tool_call_id(
                                    call.id,
                                    id_map=tool_call_id_map,
                                ),
                                function=FunctionCall(
                                    name=call.function.name,
                                    arguments=to_json_object_or_string(
                                        call.function.arguments
                                    ),
                                ),
                            )
                            for call in message.tool_calls
                        ]

                    prompt.append(assistant_msg)
                case Role.FUNCTION:
                    if message.name is None:
                        raise ValidationError(
                            "Function message name must be present"
                        )
                    if pending_legacy_function_call_ids:
                        tool_call_id = pending_legacy_function_call_ids.pop(0)
                    else:
                        # Preserve backward compatibility for malformed chat histories
                        # that provide function results without a prior assistant call.
                        tool_call_id = _legacy_function_tool_call_id(
                            function_call_idx
                        )
                        function_call_idx += 1
                    prompt.append(
                        ToolMessage(
                            content=await cls._merge_content_with_attachments(
                                message,
                                file_storage=file_storage,
                            ),
                            tool_call_id=tool_call_id,
                            name=message.name,
                        )
                    )
                case Role.TOOL:
                    if message.tool_call_id is None:
                        raise ValidationError(
                            "Tool message tool_call_id must be present"
                        )
                    tool_content = await cls._merge_content_with_attachments(
                        message,
                        file_storage=file_storage,
                    )
                    prompt.append(
                        ToolMessage(
                            content=tool_content,
                            tool_call_id=to_mistral_tool_call_id(
                                message.tool_call_id,
                                id_map=tool_call_id_map,
                            ),
                            # Preserve incoming payload as-is. Do not infer `name`
                            # from tool IDs because vanilla Mistral tool-result
                            # examples commonly omit it.
                            name=message.name,
                        )
                    )
                case _:
                    assert_never(message.role)

        return prompt

    @staticmethod
    def _to_mistral_content(
        content: str | list[MessageContentPart] | None,
    ) -> str | list[ContentChunk]:
        match content:
            case None:
                return ""
            case str():
                return content
            case list():
                chunks = []
                for part in content:
                    match part:
                        case MessageContentTextPart(text=text):
                            chunks.append(TextChunk(text=text))
                        case MessageContentImagePart(image_url=image_url):
                            chunks.append(
                                ImageURLChunk(
                                    image_url=ImageURL(url=image_url.url)
                                )
                            )
                        case MessageContentFilePart():
                            raise ValidationError(
                                "File content parts aren't supported for GCP Mistral models"
                            )
                        case MessageContentAudioPart():
                            raise ValidationError(
                                "Audio content parts aren't supported for GCP Mistral models"
                            )
                        case MessageContentRefusalPart():
                            raise ValidationError(
                                "Can't extract text from a refusal content part"
                            )
                        case _:
                            assert_never(part)

                return chunks
            case _:
                assert_never(content)

    @classmethod
    async def _merge_content_with_attachments(
        cls,
        message: Message,
        *,
        file_storage: FileStorage | None,
    ) -> str | list[ContentChunk]:
        base_content = cls._to_mistral_content(message.content)
        attachment_chunks = await cls._to_attachment_chunks(
            message, file_storage
        )
        if not attachment_chunks:
            return base_content
        if isinstance(base_content, str):
            if base_content == "":
                base_content = []
            else:
                base_content = [TextChunk(text=base_content)]
        return [*base_content, *attachment_chunks]

    @staticmethod
    async def _to_attachment_chunks(
        message: Message,
        file_storage: FileStorage | None,
    ) -> list[ImageURLChunk]:
        chunks: list[ImageURLChunk] = []
        for attachment in get_attachments(message):
            resource = await AttachmentResource(
                attachment=attachment,
                entity_name="attachment",
            ).download(file_storage)

            if not resource.type.startswith("image/"):
                raise ValidationError(
                    f"Attachment of type {resource.type!r} aren't supported"
                )
            chunks.append(
                ImageURLChunk(image_url=ImageURL(url=resource.to_data_url()))
            )
        return chunks


def _legacy_function_tool_call_id(index: int) -> str:
    # Mistral requires tool_call_id to be exactly 9 alphanumeric chars.
    return f"fc{index:07d}"


def to_mistral_tool_call_id(value: str, *, id_map: dict[str, str]) -> str:
    if (mapped := id_map.get(value)) is not None:
        return mapped

    if _MISTRAL_TOOL_CALL_ID_PATTERN.fullmatch(value):
        id_map[value] = value
        return value

    mapped = f"tc{len(id_map):07d}"
    id_map[value] = mapped
    return mapped


def _normalize_tool_parameters(parameters: dict | None) -> dict | None:
    if not isinstance(parameters, dict):
        return parameters
    # Mistral GCP tool validation rejects JSON schemas that use local
    # references ($ref to $defs), so we inline those refs before sending.
    try:
        return inline_local_json_refs(parameters)
    except ValueError as e:
        raise ValidationError(str(e)) from e
