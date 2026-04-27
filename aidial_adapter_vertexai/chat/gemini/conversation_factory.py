import json
from typing import assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion.request import Role
from google.genai.types import Content as GenAIContent
from google.genai.types import (
    FunctionResponse,
    FunctionResponseBlob,
    FunctionResponsePart,
)
from google.genai.types import Part as GenAIPart

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
    Parts,
)
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.state import update_with_message_state
from aidial_adapter_vertexai.utils.resource import Resource

GeminiConversationGenAI = BaseConversation[list[GenAIPart], GenAIContent]


class ConversationFactoryGenAI(
    ConversationFactoryBase[GenAIPart, GenAIContent, GeminiConversationGenAI],
):
    @staticmethod
    def to_gemini_genai_role(role: Role) -> str:
        match role:
            case Role.SYSTEM:
                raise ValidationError(
                    "System messages other than the first system message are not allowed"
                )
            case Role.DEVELOPER:
                raise ValidationError(
                    "Developer messages other than the first developer message are not allowed"
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
        return GenAIPart.from_text(text=text)

    def create_function_call_part(
        self, name: str, args: str, tool_call_id: str
    ) -> GenAIPart:
        try:
            return GenAIPart.from_function_call(
                name=name, args=json.loads(args)
            )
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            ) from None

    def create_function_result_part(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_call_result: str,
        resources: list[Resource],
    ) -> GenAIPart:
        try:
            processed_args = json.loads(tool_call_result)
        except Exception:
            processed_args = tool_call_result

        if isinstance(processed_args, dict):
            response = processed_args
        else:
            response = {"output": processed_args}

        response_parts = [
            FunctionResponsePart(
                inline_data=FunctionResponseBlob(
                    data=resource.data,
                    mime_type=resource.type,
                )
            )
            for resource in resources
        ]

        function_response = FunctionResponse(
            name=tool_name,
            id=tool_call_id,
            response=response,
            parts=response_parts,
        )
        return GenAIPart(function_response=function_response)

    def create_content(
        self, idx: int, dial_message: DialMessage, parts: Parts[GenAIPart]
    ) -> GenAIContent:
        role = self.to_gemini_genai_role(dial_message.role)
        content = GenAIContent(role=role, parts=parts.parts)
        update_with_message_state(idx, dial_message, content)
        return content

    def create_conversation(
        self,
        system_instruction: list[GenAIPart] | None,
        contents: list[GenAIContent],
    ) -> GeminiConversationGenAI:
        return BaseConversation.create(
            system=system_instruction, messages=contents
        )
