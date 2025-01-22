from typing import Any, List, Tuple, assert_never

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
    ConversationT,
    PartT,
)
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessorsBase,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig

FunctionName = str
FunctionArgs = str


async def messages_to_conversation(
    conversation_factory: ConversationFactoryBase[PartT, Any, ConversationT],
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    messages: List[Message],
) -> ConversationT:
    message_parts = [
        (
            message.role,
            await _message_to_parts(
                processors, tools, message, conversation_factory
            ),
        )
        for message in messages
    ]

    system, message_parts = _separate_system_messages(message_parts)

    contents = [
        conversation_factory.create_content(role, parts)
        for role, parts in message_parts
    ]

    return conversation_factory.create_conversation(system, contents)


async def _message_to_parts(
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
                        call.function.name,
                        call.function.arguments,
                        tool_call_id=call.id,
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
                conversation_factory.create_function_result_part(
                    name, content, tool_call_id=tool_call_id
                )
            ]

        case _:
            assert_never(message.role)


def _separate_system_messages(
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
