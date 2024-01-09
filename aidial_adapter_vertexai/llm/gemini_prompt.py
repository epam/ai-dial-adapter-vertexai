from typing import List

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.llm.exceptions import ValidationError


class GeminiPrompt(BaseModel):
    history: List[Content]
    prompt: List[Part]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def parse(cls, messages: List[Message]) -> "GeminiPrompt":
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        history = [parse_content(message) for message in messages]
        return cls(history=history[:-1], prompt=history[-1].parts)

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]


def parse_content(message: Message) -> Content:
    content = message.content
    if content is None:
        raise ValueError("Message content must be present")

    parts = [Part.from_text(content)]

    match message.role:
        case Role.SYSTEM:
            raise ValidationError(
                "System messages are not allowed in Gemini models"
            )
        case Role.USER:
            return Content(role=ChatSession._USER_ROLE, parts=parts)
        case Role.ASSISTANT:
            return Content(role=ChatSession._MODEL_ROLE, parts=parts)
        case Role.FUNCTION:
            raise ValidationError("Function messages are not supported")
