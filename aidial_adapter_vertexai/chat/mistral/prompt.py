import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, assert_never

from aidial_sdk.chat_completion import (
    InputFile,
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
from mistralai.gcp.client.models import (
    AssistantMessage,
    Function,
    FunctionCall,
    FunctionName,
    ImageURL,
    ImageURLChunk,
    JSONSchema,
    ResponseFormat,
    SystemMessage,
    TextChunk,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import (
    ModelParameters,
    get_attachments,
    is_system_role,
)
from aidial_adapter_vertexai.dial_api.resource import AttachmentResource
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.resource import Resource

_EMPTY_OBJECT_JSON_SCHEMA = {"type": "object", "properties": {}}
_MISTRAL_TOOL_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{9}$")
_DEFAULT_TOOL_DESCRIPTION = "Tool function"


@dataclass
class MistralPrompt:
    messages: list[Any]
    tools: list[Tool] | None = None
    use_tool_api: bool = True
    tool_choice: (
        Literal["auto", "none", "any", "required"] | ToolChoice | None
    ) = None
    response_format: ResponseFormat | None = None


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
                raise ValidationError(
                    f"Unsupported response_format for Mistral: {type(format_)}"
                )

    @staticmethod
    def _to_mistral_tools(tools: ToolsConfig) -> list[Tool] | None:
        if tools.is_empty():
            return None

        return [
            Tool(
                function=Function(
                    name=tool.function.name,
                    description=_normalize_tool_description(
                        tool.function.description
                    ),
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
    ) -> Literal["auto", "none", "any", "required"] | ToolChoice | None:
        if tools.is_empty():
            return None

        choice = tools.tool_choice
        if isinstance(choice, str):
            if choice == "required":
                return "required"
            if choice in ("auto", "none"):
                return choice
            raise ValidationError(f"Unsupported tool_choice value: {choice!r}")
        if isinstance(choice, DialToolChoice):
            return ToolChoice(function=FunctionName(name=choice.function.name))
        raise ValidationError(f"Unsupported tool_choice value: {choice!r}")

    @classmethod
    async def _to_mistral_messages(
        cls,
        messages: list[Message],
        file_storage: FileStorage | None,
    ) -> list[Any]:
        prompt: list[Any] = []
        function_call_idx = 0
        pending_legacy_function_call_ids: list[str] = []
        tool_call_id_map: dict[str, str] = {}

        for message in messages:
            if is_system_role(message.role):
                prompt.append(
                    SystemMessage(
                        content=await cls._merge_content_with_attachments(
                            message,
                            file_storage=file_storage,
                            role_content_name="System message content",
                            allow_images=False,
                        )
                    )
                )
                continue

            if message.role == Role.USER:
                prompt.append(
                    UserMessage(
                        content=await cls._merge_content_with_attachments(
                            message,
                            file_storage=file_storage,
                            role_content_name="User message content",
                            allow_images=True,
                        )
                    )
                )
                continue

            if message.role == Role.ASSISTANT:
                if message.function_call is not None:
                    tool_call_id = _legacy_function_tool_call_id(
                        function_call_idx
                    )
                    function_call_idx += 1
                    pending_legacy_function_call_ids.append(tool_call_id)
                    function_call = FunctionCall(
                        name=message.function_call.name,
                        arguments=_to_json_object_or_string(
                            message.function_call.arguments or ""
                        ),
                    )
                    prompt.append(
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id=tool_call_id, function=function_call
                                )
                            ],
                        )
                    )
                    continue

                if message.tool_calls:
                    prompt.append(
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id=_to_mistral_tool_call_id(
                                        call.id,
                                        id_map=tool_call_id_map,
                                    ),
                                    function=FunctionCall(
                                        name=call.function.name,
                                        arguments=_to_json_object_or_string(
                                            call.function.arguments or ""
                                        ),
                                    ),
                                )
                                for call in message.tool_calls
                            ],
                        )
                    )
                    continue

                prompt.append(
                    AssistantMessage(
                        content=_normalize_empty_regular_message(
                            await cls._merge_content_with_attachments(
                                message,
                                file_storage=file_storage,
                                role_content_name="Assistant message content",
                                allow_images=True,
                            )
                        )
                    )
                )
                continue

            if message.role == Role.FUNCTION:
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
                            role_content_name="Function message content",
                            allow_images=True,
                        ),
                        tool_call_id=tool_call_id,
                        name=message.name,
                    )
                )
                continue

            if message.role == Role.TOOL:
                if message.tool_call_id is None:
                    raise ValidationError(
                        "Tool message tool_call_id must be present"
                    )
                tool_content = await cls._merge_content_with_attachments(
                    message,
                    file_storage=file_storage,
                    role_content_name="Tool message content",
                    allow_images=True,
                )
                prompt.append(
                    ToolMessage(
                        content=tool_content,
                        tool_call_id=_to_mistral_tool_call_id(
                            message.tool_call_id,
                            id_map=tool_call_id_map,
                        ),
                        # Preserve incoming payload as-is. Do not infer `name`
                        # from tool IDs because vanilla Mistral tool-result
                        # examples commonly omit it.
                        name=message.name,
                    )
                )
                continue

            raise ValidationError(
                f"Role {message.role.value!r} isn't supported for Mistral models"
            )

        return prompt

    @staticmethod
    def _to_mistral_content(
        content: str | list[MessageContentPart] | None,
        *,
        allow_images: bool,
        content_name: str,
    ) -> Any:
        match content:
            case None:
                return ""
            case str():
                return content
            case list():
                chunks: list[Any] = []
                text_parts: list[str] = []
                has_media = False

                for part in content:
                    match part:
                        case MessageContentTextPart(text=text):
                            text_parts.append(text)
                            chunks.append(TextChunk(text=text))
                        case MessageContentImagePart(image_url=image_url):
                            if not allow_images:
                                raise ValidationError(
                                    f"{content_name} doesn't support image parts"
                                )
                            has_media = True
                            chunks.append(
                                ImageURLChunk(
                                    image_url=ImageURL(url=image_url.url)
                                )
                            )
                        case MessageContentFilePart():
                            file_chunk = _file_content_part_to_image_url_chunk(
                                part.file,
                                content_name=content_name,
                            )
                            has_media = True
                            chunks.append(file_chunk)
                        case MessageContentAudioPart():
                            raise ValidationError(
                                "Audio content parts aren't supported for Mistral models"
                            )
                        case MessageContentRefusalPart():
                            raise ValidationError(
                                "Can't extract text from a refusal content part"
                            )
                        case _:
                            assert_never(part)

                if has_media:
                    return chunks
                return "\n\n".join(text_parts)
            case _:
                assert_never(content)

    @classmethod
    async def _merge_content_with_attachments(
        cls,
        message: Message,
        *,
        file_storage: FileStorage | None,
        role_content_name: str,
        allow_images: bool,
    ) -> Any:
        base_content = cls._to_mistral_content(
            message.content,
            allow_images=allow_images,
            content_name=role_content_name,
        )
        attachment_chunks = await cls._to_attachment_chunks(
            message, file_storage, allow_images=allow_images
        )

        if not attachment_chunks:
            return base_content

        if isinstance(base_content, str):
            chunks: list[Any] = (
                [TextChunk(text=base_content)] if base_content else []
            )
        elif isinstance(base_content, list):
            chunks = list(base_content)
        else:
            chunks = []

        chunks.extend(attachment_chunks)
        return chunks

    @staticmethod
    async def _to_attachment_chunks(
        message: Message,
        file_storage: FileStorage | None,
        *,
        allow_images: bool,
    ) -> list[ImageURLChunk]:
        if not allow_images:
            if get_attachments(message):
                raise ValidationError(
                    "System message content doesn't support attachments"
                )
            return []

        chunks: list[ImageURLChunk] = []
        for attachment in get_attachments(message):
            resource = await AttachmentResource(
                attachment=attachment,
                entity_name="attachment",
            ).download(file_storage)

            if (
                resource.type.startswith("image/")
                or resource.type == "application/pdf"
            ):
                chunks.append(
                    ImageURLChunk(
                        image_url=ImageURL(url=resource.to_data_url())
                    )
                )
        return chunks


def _to_json_object_or_string(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    try:
        return json.loads(value)
    except ValueError:
        return value


def _file_content_part_to_image_url_chunk(
    file: InputFile, *, content_name: str
) -> ImageURLChunk:
    if (file_data := file.file_data) is None:
        raise ValidationError(
            f"{content_name}: file content part must have file_data field"
        )

    resource = Resource.from_data_url(file_data)
    if resource is None:
        # Keep compatibility with existing adapters/tests that often pass
        # base64-encoded PDFs in file_data without data URL prefix.
        try:
            resource = Resource.from_base64("application/pdf", file_data)
        except Exception:
            raise ValidationError(
                f"{content_name}: invalid file_data in file content part"
            ) from None

    if not (
        resource.type.startswith("image/") or resource.type == "application/pdf"
    ):
        raise ValidationError(
            f"{content_name}: unsupported file content part type {resource.type!r}"
        )

    return ImageURLChunk(image_url=ImageURL(url=resource.to_data_url()))


def _normalize_empty_regular_message(content: Any) -> Any:
    """
    Some upstreams reject assistant messages when content is effectively empty.
    Keep protocol compatibility by substituting a single-space placeholder.
    """
    return " " if content == "" else content


def _legacy_function_tool_call_id(index: int) -> str:
    # Mistral requires tool_call_id to be exactly 9 alphanumeric chars.
    return f"fc{index:07d}"


def _to_mistral_tool_call_id(value: str, *, id_map: dict[str, str]) -> str:
    if (mapped := id_map.get(value)) is not None:
        return mapped

    if _MISTRAL_TOOL_CALL_ID_PATTERN.fullmatch(value):
        id_map[value] = value
        return value

    mapped = f"tc{len(id_map):07d}"
    id_map[value] = mapped
    return mapped


def _normalize_tool_description(description: str | None) -> str:
    # Mistral tool calling becomes unreliable when description is empty/missing.
    if description is None:
        return _DEFAULT_TOOL_DESCRIPTION
    normalized = description.strip()
    if normalized == "":
        return _DEFAULT_TOOL_DESCRIPTION
    return description


def _normalize_tool_parameters(parameters: Any) -> Any:
    if not isinstance(parameters, dict):
        return parameters
    # Mistral GCP tool validation rejects JSON schemas that use local
    # references ($ref to $defs), so we inline those refs before sending.
    return _resolve_local_json_refs(parameters)


def _resolve_local_json_refs(schema: dict[str, Any]) -> dict[str, Any]:
    root = deepcopy(schema)

    def _resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/", 2)[-1]
                defs = root.get("$defs", {})
                target = defs.get(key)
                if isinstance(target, dict):
                    resolved_target = _resolve(deepcopy(target))
                    # Keep sibling constraints while replacing the ref.
                    siblings = {
                        k: _resolve(v) for k, v in node.items() if k != "$ref"
                    }
                    if isinstance(resolved_target, dict):
                        return {**resolved_target, **siblings}
        return {k: _resolve(v) for k, v in node.items()}

    normalized = _resolve(root)
    if isinstance(normalized, dict):
        normalized.pop("$defs", None)
    return normalized if isinstance(normalized, dict) else schema
