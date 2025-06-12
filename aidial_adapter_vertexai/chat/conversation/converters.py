from typing import Any, List, Tuple, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from aidial_adapter_vertexai.chat.attachment_processor import (
    AttachmentProcessorsBase,
)
from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
    ConversationT,
    Parts,
    PartT,
)
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import is_system_role

FunctionName = str
FunctionArgs = str


class Counter:
    count: int = 0

    def post_inc(self):
        old_count = self.count
        self.count += 1
        return old_count


async def messages_to_conversation(
    conversation_factory: ConversationFactoryBase[PartT, Any, ConversationT],
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    messages: List[DialMessage],
) -> ConversationT:
    function_call_idx = Counter()

    message_parts = [
        (
            message,
            await _message_to_parts(
                processors,
                tools,
                message,
                conversation_factory,
                function_call_idx,
            ),
        )
        for message in messages
    ]

    system, message_parts = _separate_system_messages(message_parts)

    contents = [
        conversation_factory.create_content(dial_message, parts)
        for dial_message, parts in message_parts
    ]

    return conversation_factory.create_conversation(system, contents)


async def _message_to_parts(
    processors: AttachmentProcessorsBase[PartT],
    tools: ToolsConfig,
    message: DialMessage,
    conversation_factory: ConversationFactoryBase,
    function_call_idx: Counter,
) -> Parts[PartT]:
    content = message.content

    match message.role:
        case Role.SYSTEM:
            if content is None:
                raise ValidationError("System message content must be present")
            return await processors.process_message(message)
        case Role.DEVELOPER:
            if content is None:
                raise ValidationError(
                    "Developer message content must be present"
                )
            return await processors.process_message(message)
        case Role.USER:
            if not content:
                raise ValidationError("User message content must be present")
            return await processors.process_message(message)

        case Role.ASSISTANT:
            if message.function_call is not None:
                tool_call_id = f"function_call_{function_call_idx.count}"
                return Parts(
                    parts=[
                        conversation_factory.create_function_call_part(
                            message.function_call.name,
                            message.function_call.arguments,
                            tool_call_id,
                        )
                    ]
                )
            elif message.tool_calls is not None:
                return Parts(
                    parts=[
                        conversation_factory.create_function_call_part(
                            call.function.name,
                            call.function.arguments,
                            call.id,
                        )
                        for call in message.tool_calls
                    ]
                )
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
            tool_call_id = f"function_call_{function_call_idx.post_inc()}"
            return Parts(
                parts=[
                    conversation_factory.create_function_result_part(
                        name, content, tool_call_id
                    )
                ]
            )

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
            return Parts(
                parts=[
                    conversation_factory.create_function_result_part(
                        name, content, tool_call_id
                    )
                ]
            )

        case _:
            assert_never(message.role)


def _separate_system_messages(
    messages: List[Tuple[DialMessage, Parts[PartT]]],
) -> Tuple[List[PartT] | None, List[Tuple[DialMessage, Parts[PartT]]]]:
    """
    Extract the leading system messages from the list of messages.
    """
    if len(messages) == 0:
        return None, messages

    system_messages: List[PartT] = []

    while messages:
        dial_message, message = messages[0]
        if is_system_role(dial_message.role):
            if message.resources:
                raise ValidationError(
                    "System messages cannot contain attachments"
                )
            system_messages.extend(message.parts)
            messages = messages[1:]
        else:
            break

    return system_messages or None, messages
