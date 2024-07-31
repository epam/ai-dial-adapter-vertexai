import json
from typing import Any, Dict, List, assert_never

from aidial_sdk.chat_completion import FunctionCall, Message, Role, ToolCall
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.utils.resource import Resource


class MessageWithResources(BaseModel):
    message: Message
    resources: List[Resource]

    @property
    def content(self) -> str:
        content = self.message.content
        if content is None:
            raise ValidationError("Message content must be present")

        # Gemini doesn't support empty messages: neither user's nor assistant's.
        # It throws an error:
        #   400 Unable to submit request because it has an empty text parameter.
        #   Add a value to the parameter and try again.
        if content == "":
            content = " "

        return content

    def to_text(self) -> str:
        if len(self.resources) > 0:
            raise ValidationError("Inputs are not supported for text messages")

        return self.content

    def to_parts(self, tools: ToolsConfig) -> List[Part]:
        # Placing Images/Video parts before the text as per
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?authuser=1#image_best_practices
        parts = [resource.to_part() for resource in self.resources]
        parts.extend(message_to_gemini(tools, self.message))

        return parts

    def to_content(self, tools: ToolsConfig) -> Content:
        return Content(
            role=from_dial_role(self.message.role),
            parts=self.to_parts(tools),
        )


def from_dial_role(role: Role) -> str:
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


def message_to_gemini(tools: ToolsConfig, message: Message) -> List[Part]:

    content = message.content

    match message.role:
        case Role.SYSTEM:
            if content is None:
                raise ValidationError("System message content must be present")
            return [Part.from_text(content)]

        case Role.USER:
            if content is None:
                raise ValidationError("User message content must be present")
            return [Part.from_text(content)]

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
                return [Part.from_text(content)]

        case Role.FUNCTION:
            if content is None:
                raise ValidationError(
                    "Function message content must be present"
                )
            args = content_to_function_args(content)
            name = message.name
            if name is None:
                raise ValidationError("Function message name must be present")
            return [Part.from_function_response(name, args)]

        case Role.TOOL:
            if content is None:
                raise ValidationError("Tool message content must be present")
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
