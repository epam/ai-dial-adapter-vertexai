import json
from typing import Callable, List, Tuple, TypeVar, Union, assert_never

from aidial_sdk.chat_completion import Message, Role
from google.genai.types import Content as GenAiContent
from google.genai.types import Part as GenAiPart
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessors,
    AttachmentProcessorsBase,
    AttachmentProcessorsGenAI,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import (
    ContentT,
    GeminiConversation,
    GeminiGenAIConversation,
    PartT,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig

FunctionName = str
FunctionArgs = str


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


def _to_gemini_genai_role(role: Role) -> str:
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


GeminiConversationT = TypeVar(
    "GeminiConversationT",
    bound=Union[GeminiConversation, GeminiGenAIConversation],
)


async def messages_to_gemini_conversation_base(
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    messages: List[Message],
    content_factory: Callable[[Role, List[PartT]], ContentT],
    conversation_factory: Callable[
        [List[PartT] | None, List[ContentT]], GeminiConversationT
    ],
    function_call_factory: Callable[[FunctionName, FunctionArgs], PartT],
    function_result_factory: Callable[[FunctionName, str], PartT],
) -> GeminiConversationT:
    gemini_messages = [
        (
            await _message_to_gemini_parts(
                processors,
                tools,
                message,
                function_call_factory,
                function_result_factory,
            ),
            message.role,
        )
        for message in messages
    ]

    system_instruction, gemini_messages = separate_system_messages(
        gemini_messages
    )

    contents = [content_factory(role, parts) for parts, role in gemini_messages]

    return conversation_factory(
        system_instruction,
        contents,
    )


async def messages_to_gemini_conversation(
    processors: AttachmentProcessors,
    tools: ToolsConfig,
    messages: List[Message],
) -> GeminiConversation:
    def _content_factory(role: Role, parts: List[Part]) -> Content:
        return Content(role=_to_gemini_role(role), parts=parts)

    def _conversation_factory(
        system_instruction: List[Part] | None, contents: List[Content]
    ) -> GeminiConversation:
        return GeminiConversation(
            system_instruction=system_instruction, contents=contents
        )

    def _function_call_factory(name: FunctionName, args: FunctionArgs) -> Part:
        try:
            args = json.loads(args)
            return Part.from_dict(
                {"function_call": {"name": name, "args": args}}
            )
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            )

    def _function_result_factory(name: str, args: str) -> Part:
        try:
            args = json.loads(args)
        except Exception:
            args = args

        if isinstance(args, dict):
            return Part.from_function_response(name, args)

        return Part.from_function_response(name, {"content": args})

    return await messages_to_gemini_conversation_base(
        processors,
        tools,
        messages,
        _content_factory,
        _conversation_factory,
        _function_call_factory,
        _function_result_factory,
    )


async def messages_to_gemini_genai_conversation(
    processors: AttachmentProcessorsGenAI,
    tools: ToolsConfig,
    messages: List[Message],
) -> GeminiGenAIConversation:
    def _content_factory(role: Role, parts: List[GenAiPart]) -> GenAiContent:
        return GenAiContent(role=_to_gemini_genai_role(role), parts=parts)

    def _conversation_factory(
        system_instruction: List[GenAiPart] | None, contents: List[GenAiContent]
    ) -> GeminiGenAIConversation:
        return GeminiGenAIConversation(
            system_instruction=system_instruction, contents=contents
        )

    def _function_call_factory(
        name: FunctionName, args: FunctionArgs
    ) -> GenAiPart:
        try:
            return GenAiPart.from_function_call(name, json.loads(args))
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            )

    def _function_result_factory(name: str, content: str) -> GenAiPart:
        try:
            args = json.loads(content)
        except Exception:
            args = content

        if isinstance(args, dict):
            return GenAiPart.from_function_response(name, args)

        return GenAiPart.from_function_response(name, {"output": args})

    return await messages_to_gemini_conversation_base(
        processors,
        tools,
        messages,
        _content_factory,
        _conversation_factory,
        _function_call_factory,
        _function_result_factory,
    )


async def _message_to_gemini_parts(
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    message: Message,
    function_call_factory: Callable[[FunctionName, FunctionArgs], PartT],
    function_response_factory: Callable[[FunctionName, FunctionArgs], PartT],
) -> List[PartT]:

    content = message.content

    match message.role:
        case Role.SYSTEM:
            if content is None:
                raise ValidationError("System message content must be present")
            return await processors.process_message(message)

        case Role.USER:
            if not content:
                raise ValidationError("User message content must be present")
            return await processors.process_message(message)

        case Role.ASSISTANT:
            if message.function_call is not None:
                return [
                    function_call_factory(
                        message.function_call.name,
                        message.function_call.arguments,
                    )
                ]
            elif message.tool_calls is not None:
                return [
                    function_call_factory(
                        call.function.name, call.function.arguments
                    )
                    for call in message.tool_calls
                ]
            else:
                if not content:
                    raise ValidationError(
                        "Assistant message content must be present"
                    )
                return await processors.process_message(message)

        case Role.FUNCTION:
            if content is None:
                raise ValidationError(
                    "Function message content must be present"
                )
            if not isinstance(content, str):
                raise ValidationError(
                    "Function message content must be a string"
                )
            name = message.name
            if name is None:
                raise ValidationError("Function message name must be present")
            return [function_response_factory(name, content)]

        case Role.TOOL:
            if content is None:
                raise ValidationError("Tool message content must be present")
            if not isinstance(content, str):
                raise ValidationError("Tool message content must be a string")
            tool_call_id = message.tool_call_id
            if tool_call_id is None:
                raise ValidationError(
                    "Tool message tool_call_id must be present"
                )
            name = tools.get_tool_name(tool_call_id)
            return [function_response_factory(name, content)]

        case _:
            assert_never(message.role)


def separate_system_messages(
    messages: List[Tuple[List[PartT], Role]]
) -> Tuple[List[PartT] | None, List[Tuple[List[PartT], Role]]]:
    """
    Extract the leading system messages from the list of messages.
    """
    if len(messages) == 0:
        return None, messages

    system_messages: List[PartT] = []

    while messages:
        message, role = messages[0]
        if role == Role.SYSTEM:
            system_messages.extend(message)
            messages = messages[1:]
        else:
            break

    return system_messages or None, messages
