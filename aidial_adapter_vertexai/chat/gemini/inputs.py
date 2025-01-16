from typing import Any, List, Tuple, assert_never

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.conversation_factory import (
    ConversationFactory,
    ConversationFactoryBase,
    GenAIConversationFactory,
)
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessors,
    AttachmentProcessorsBase,
    AttachmentProcessorsGenAI,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import (
    GeminiConversation,
    GeminiConversationT,
    GeminiGenAIConversation,
    PartT,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig

FunctionName = str
FunctionArgs = str


async def messages_to_gemini_conversation_base(
    conversation_factory: ConversationFactoryBase[
        PartT, Any, GeminiConversationT
    ],
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    messages: List[Message],
) -> GeminiConversationT:
    gemini_messages = [
        (
            message.role,
            await _message_to_gemini_parts(
                processors, tools, message, conversation_factory
            ),
        )
        for message in messages
    ]

    system_instruction, gemini_messages = separate_system_messages(
        gemini_messages
    )

    contents = [
        conversation_factory.create_content(role, parts)
        for role, parts in gemini_messages
    ]

    return conversation_factory.create_conversation(
        system_instruction,
        contents,
    )


async def messages_to_gemini_conversation(
    conversation_factory: ConversationFactory,
    processors: AttachmentProcessors,
    tools: ToolsConfig,
    messages: List[Message],
) -> GeminiConversation:

    return await messages_to_gemini_conversation_base(
        conversation_factory,
        processors,
        tools,
        messages,
    )


async def messages_to_gemini_genai_conversation(
    conversation_factory: GenAIConversationFactory,
    processors: AttachmentProcessorsGenAI,
    tools: ToolsConfig,
    messages: List[Message],
) -> GeminiGenAIConversation:

    return await messages_to_gemini_conversation_base(
        conversation_factory,
        processors,
        tools,
        messages,
    )


async def _message_to_gemini_parts(
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    message: Message,
    conversation_factory: ConversationFactoryBase,
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
                    conversation_factory.create_function_call_part(
                        message.function_call.name,
                        message.function_call.arguments,
                    )
                ]
            elif message.tool_calls is not None:
                return [
                    conversation_factory.create_function_call_part(
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
            return [
                conversation_factory.create_function_result_part(name, content)
            ]

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
            return [
                conversation_factory.create_function_result_part(name, content)
            ]

        case _:
            assert_never(message.role)


def separate_system_messages(
    messages: List[Tuple[Role, List[PartT]]]
) -> Tuple[List[PartT] | None, List[Tuple[Role, List[PartT]]]]:
    """
    Extract the leading system messages from the list of messages.
    """
    if len(messages) == 0:
        return None, messages

    system_messages: List[PartT] = []

    while messages:
        role, message = messages[0]
        if role == Role.SYSTEM:
            system_messages.extend(message)
            messages = messages[1:]
        else:
            break

    return system_messages or None, messages
