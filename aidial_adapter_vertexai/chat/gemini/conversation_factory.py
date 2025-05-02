import json
from typing import Generic, List, assert_never

from aidial_sdk.chat_completion.request import Role
from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from pydantic.v1 import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.conversation.factory import (
    ContentT,
    ConversationFactoryBase,
    PartT,
)
from aidial_adapter_vertexai.chat.errors import ValidationError


class GeminiConversationBase(BaseModel, Generic[PartT, ContentT]):
    system_instruction: List[PartT] | None = None
    contents: List[ContentT]

    class Config:
        arbitrary_types_allowed = True


class GeminiConversation(GeminiConversationBase[Part, Content]):
    pass


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

    def create_content(self, role: Role, parts: List[Part]) -> Content:
        return Content(role=self._to_gemini_role(role), parts=parts)

    def create_conversation(
        self, system_instruction: List[Part] | None, contents: List[Content]
    ) -> GeminiConversation:
        return GeminiConversation(
            system_instruction=system_instruction, contents=contents
        )


class GeminiGenAIConversation(GeminiConversationBase[GenAIPart, GenAIContent]):
    pass


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
        return GenAIPart.from_text(text)

    def create_function_call_part(
        self, name: str, args: str, tool_call_id: str
    ) -> GenAIPart:
        try:
            return GenAIPart.from_function_call(name, json.loads(args))
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
            return GenAIPart.from_function_response(name, processed_args)

        return GenAIPart.from_function_response(
            name, {"output": processed_args}
        )

    def create_content(
        self, role: Role, parts: List[GenAIPart]
    ) -> GenAIContent:
        return GenAIContent(role=self.to_gemini_genai_role(role), parts=parts)

    def create_conversation(
        self,
        system_instruction: List[GenAIPart] | None,
        contents: List[GenAIContent],
    ) -> GeminiGenAIConversation:
        return GeminiGenAIConversation(
            system_instruction=system_instruction, contents=contents
        )
