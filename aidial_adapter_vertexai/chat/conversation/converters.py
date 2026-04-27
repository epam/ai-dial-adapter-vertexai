from typing import Any, assert_never

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
    messages: list[DialMessage],
) -> ConversationT:
    function_call_idx = Counter()

    message_parts = [
        (
            idx,
            message,
            await _message_to_parts(
                processors,
                tools,
                message,
                conversation_factory,
                function_call_idx,
            ),
        )
        for (idx, message) in enumerate(messages)
    ]

    system, message_parts = _separate_system_messages(message_parts)

    contents = [
        conversation_factory.create_content(idx, dial_message, parts)
        for idx, dial_message, parts in message_parts
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
    content_parts = await processors.process_message(message)

    role = message.role.value.capitalize()

    match message.role:
        case Role.SYSTEM | Role.DEVELOPER | Role.USER:
            if content is None:
                raise ValidationError(f"{role} message content must be present")
            return content_parts

        case Role.ASSISTANT:
            if message.function_call is not None:
                tool_call_id = f"function_call_{function_call_idx.count}"
                content_parts.append_part(
                    conversation_factory.create_function_call_part(
                        message.function_call.name,
                        message.function_call.arguments,
                        tool_call_id,
                    )
                )

            if message.tool_calls is not None:
                content_parts.append_parts(
                    [
                        conversation_factory.create_function_call_part(
                            call.function.name,
                            call.function.arguments,
                            call.id,
                        )
                        for call in message.tool_calls
                    ]
                )

            return content_parts

        case Role.FUNCTION | Role.TOOL:
            if content is None:
                raise ValidationError(f"{role} message content must be present")
            if not isinstance(content, str):
                raise ValidationError(
                    f"{role} message content must be a string"
                )

            if message.role == Role.FUNCTION:
                tool_call_id = f"function_call_{function_call_idx.post_inc()}"
                tool_name = message.name
                if tool_name is None:
                    raise ValidationError(
                        f"{role} message name must be present"
                    )
            else:
                tool_call_id = message.tool_call_id
                if tool_call_id is None:
                    raise ValidationError(
                        f"{role} message tool_call_id must be present"
                    )
                tool_name = tools.get_tool_name(tool_call_id)

            return Parts(
                parts=[
                    conversation_factory.create_function_result_part(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_call_result=content,
                        resources=content_parts.resources,
                    )
                ]
            )

        case _:
            assert_never(message.role)


def _separate_system_messages(
    messages: list[tuple[int, DialMessage, Parts[PartT]]],
) -> tuple[list[PartT] | None, list[tuple[int, DialMessage, Parts[PartT]]]]:
    """
    Extract the leading system messages from the list of messages.
    """
    if len(messages) == 0:
        return None, messages

    system_messages: list[PartT] = []

    while messages:
        _idx, dial_message, message = messages[0]
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
