import json
from typing import Any, Dict, List, Tuple, TypeVar, assert_never

from aidial_sdk.chat_completion import FunctionCall, Message, Role, ToolCall
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import AttachmentProcessors
from aidial_adapter_vertexai.chat.tools import ToolsConfig


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


def function_call_to_part(call: FunctionCall) -> Part:
    try:
        args = json.loads(call.arguments)
    except Exception:
        raise ValidationError("Function call arguments must be a valid JSON")
    return Part.from_dict({"function_call": {"name": call.name, "args": args}})


def tool_call_to_part(call: ToolCall) -> Part:
    return function_call_to_part(call.function)


def content_to_function_args(content: str) -> Dict[str, Any]:
    try:
        args = json.loads(content)
    except Exception:
        args = content

    if isinstance(args, dict):
        return args

    return {"content": args}


async def messages_to_gemini_history(
    processors: AttachmentProcessors,
    tools: ToolsConfig,
    messages: List[Message],
) -> List[Content]:
    gemini_messages = [
        (
            await _message_to_gemini_parts(processors, tools, message),
            message.role,
        )
        for message in messages
    ]

    gemini_messages = eliminate_leading_system_messages(gemini_messages)

    return [
        Content(role=_to_gemini_role(role), parts=parts)
        for parts, role in gemini_messages
    ]


async def _message_to_gemini_parts(
    processors: AttachmentProcessors, tools: ToolsConfig, message: Message
) -> List[Part]:

    content = message.content

    match message.role:
        case Role.SYSTEM:
            if content is None:
                raise ValidationError("System message content must be present")
            return await processors.process_message(message)

        case Role.USER:
            if content is None:
                raise ValidationError("User message content must be present")
            return await processors.process_message(message)

        case Role.ASSISTANT:
            if message.function_call is not None:
                return [function_call_to_part(message.function_call)]
            elif message.tool_calls is not None:
                return [tool_call_to_part(call) for call in message.tool_calls]
            else:
                if content is None:
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
            args = content_to_function_args(content)
            name = message.name
            if name is None:
                raise ValidationError("Function message name must be present")
            return [Part.from_function_response(name, args)]

        case Role.TOOL:
            if content is None:
                raise ValidationError("Tool message content must be present")
            if not isinstance(content, str):
                raise ValidationError("Tool message content must be a string")
            args = content_to_function_args(content)
            tool_call_id = message.tool_call_id
            if tool_call_id is None:
                raise ValidationError(
                    "Tool message tool_call_id must be present"
                )
            name = tools.get_tool_name(tool_call_id)
            return [Part.from_function_response(name, args)]

        case _:
            assert_never(message.role)


_T = TypeVar("_T")


def eliminate_leading_system_messages(
    messages: List[Tuple[List[_T], Role]]
) -> List[Tuple[List[_T], Role]]:
    """
    Attach the leading system messages to a subsequent user message.

    NOTE: it's possible to pass `system_instruction` to `GenerativeModel` constructor,
    however `system_instruction` field isn't yet fully integrated into the VertexAI SDK.
    In particular, it's not exposed in `GenerativeModel.count_tokens_async` method:
    https://github.com/googleapis/python-aiplatform/issues/3631

    NOTE: it's not enough to simply turn system message into yet another user message,
    because Gemini will complain about incorrect chat structure:
        400 Please ensure that multiturn requests alternate between user and model.
    """

    if len(messages) == 0:
        return messages

    system_messages: List[_T] = []

    while messages:
        message, role = messages[0]
        if role == Role.SYSTEM:
            system_messages.extend(message)
            messages = messages[1:]
        else:
            break

    if messages:
        message, role = messages[0]
        return [(system_messages + message, role)] + messages[1:]
    else:
        return [(system_messages, Role.USER)]
