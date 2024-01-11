from typing import List, assert_never

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.utils.list import cluster_by


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

        simple_messages = list(map(SimpleMessage.from_message, messages))
        history = [
            SimpleMessage.from_messages(cluster).to_content()
            for cluster in cluster_by(lambda c: c.role, simple_messages)
        ]

        return cls(history=history[:-1], prompt=history[-1].parts)

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]


class SimpleMessage(BaseModel):
    role: str
    content: str

    @classmethod
    def from_message(cls, message: Message) -> "SimpleMessage":
        content = message.content
        if content is None:
            raise ValueError("Message content must be present")

        match message.role:
            case Role.SYSTEM:
                role = ChatSession._USER_ROLE
            case Role.USER:
                role = ChatSession._USER_ROLE
            case Role.ASSISTANT:
                role = ChatSession._MODEL_ROLE
            case Role.FUNCTION | Role.TOOL:
                raise ValidationError("Function messages are not supported")
            case _:
                assert_never(message.role)

        return SimpleMessage(role=role, content=content)

    @classmethod
    def from_messages(cls, messages: List["SimpleMessage"]) -> "SimpleMessage":
        if len(messages) == 0:
            raise ValueError("Messages must not be empty")

        return SimpleMessage(
            role=messages[0].role,
            content="\n".join(message.content for message in messages),
        )

    def to_content(self) -> Content:
        return Content(role=self.role, parts=[Part.from_text(self.content)])
