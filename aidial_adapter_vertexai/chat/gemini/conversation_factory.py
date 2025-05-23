import json
from typing import List, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion.request import Role
from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
)
from aidial_adapter_vertexai.chat.errors import ValidationError

GeminiConversation = BaseConversation[List[Part], Content]
GeminiGenAIConversation = BaseConversation[List[GenAIPart], GenAIContent]


class ConversationFactory(
    ConversationFactoryBase[Part, Content, GeminiConversation],
):
    @staticmethod
    def _to_gemini_role(role: Role) -> str:
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
                return ChatSession._USER_ROLE
            case Role.ASSISTANT:
                return ChatSession._MODEL_ROLE
            case _:
                assert_never(role)

    def create_multi_modal_part(self, data: bytes, mime_type: str) -> Part:
        return Part.from_data(data=data, mime_type=mime_type)

    def create_text_part(self, text: str) -> Part:
        return Part.from_text(text)

    def create_function_call_part(
        self, name: str, args: str, tool_call_id: str
    ) -> Part:
        try:
            args = json.loads(args)
            return Part.from_dict(
                {"function_call": {"name": name, "args": args}}
            )
        except Exception:
            raise ValidationError(
                "Function call arguments must be a valid JSON"
            )

    def create_function_result_part(
        self, name: str, args: str, tool_call_id: str
    ) -> Part:
        try:
            args = json.loads(args)
        except Exception:
            args = args

        if isinstance(args, dict):
            return Part.from_function_response(name, args)

        return Part.from_function_response(name, {"content": args})

    def create_content(
        self, dial_message: DialMessage, parts: List[Part]
    ) -> Content:
        return Content(
            role=self._to_gemini_role(dial_message.role),
            parts=parts,
        )

    def create_conversation(
        self, system_instruction: List[Part] | None, contents: List[Content]
    ) -> GeminiConversation:
        return BaseConversation.create(
            system=system_instruction, messages=contents
        )


class GenAIConversationFactory(
    ConversationFactoryBase[GenAIPart, GenAIContent, GeminiGenAIConversation],
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
            )

    def create_function_result_part(
        self, name: str, args: str, tool_call_id: str
    ) -> GenAIPart:
        try:
            processed_args = json.loads(args)
        except Exception:
            processed_args = args

        if isinstance(processed_args, dict):
            return GenAIPart.from_function_response(
                name=name, response=processed_args
            )

        return GenAIPart.from_function_response(
            name=name, response={"output": processed_args}
        )

    def create_content(
        self, dial_message: DialMessage, parts: List[GenAIPart]
    ) -> GenAIContent:
        return GenAIContent(
            role=self.to_gemini_genai_role(dial_message.role), parts=parts
        )

    def create_conversation(
        self,
        system_instruction: List[GenAIPart] | None,
        contents: List[GenAIContent],
    ) -> GeminiGenAIConversation:
        return BaseConversation.create(
            system=system_instruction, messages=contents
        )
