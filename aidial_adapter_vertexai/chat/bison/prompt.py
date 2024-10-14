from enum import Enum
from typing import List, Optional, Tuple

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.language_models import ChatMessage, ChatSession

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.request import collect_text_content


class ChatAuthor(str, Enum):
    USER = ChatSession.USER_AUTHOR
    BOT = ChatSession.MODEL_AUTHOR

    def __repr__(self) -> str:
        return f"{self.value!r}"


class BisonPrompt(BaseModel):
    context: Optional[str]
    messages: List[ChatMessage]

    @classmethod
    def parse(cls, messages: List[Message]) -> "BisonPrompt":
        context, messages = _validate_messages_and_split(messages)
        return cls(
            context=context,
            messages=list(map(_parse_message, messages)),
        )

    @property
    def user_prompt(self) -> str:
        return self.messages[-1].content or ""

    @property
    def history(self) -> List[ChatMessage]:
        return self.messages[:-1]


_SUPPORTED_ROLES = {Role.SYSTEM, Role.USER, Role.ASSISTANT}


def _validate_messages_and_split(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message]]:
    if len(messages) == 0:
        raise ValidationError("The chat history must have at least one message")

    for message in messages:
        if message.content is None:
            raise ValidationError("Message content must be present")

        if message.role not in _SUPPORTED_ROLES:
            raise ValidationError(
                f"Message role must be one of {_SUPPORTED_ROLES}"
            )

    context: Optional[str] = None
    if len(messages) > 0 and messages[0].role == Role.SYSTEM:
        context = collect_text_content(messages[0].content)
        context = context if context.strip() else None
        messages = messages[1:]

    if len(messages) == 0 and context is not None:
        raise ValidationError(
            "The chat history must have at least one non-system message"
        )

    role: Optional[Role] = None
    for message in messages:
        if message.role == Role.SYSTEM:
            raise ValidationError(
                "System messages other than the initial system message are not allowed"
            )

        # Bison doesn't support empty messages,
        # so we replace it with a single space.
        message.content = message.content or " "

        if role == message.role:
            raise ValidationError("Messages must alternate between authors")

        role = message.role

    if len(messages) % 2 == 0:
        raise ValidationError(
            "There should be odd number of messages for correct alternating turn"
        )

    if messages[-1].role != Role.USER:
        raise ValidationError("The last message must be a user message")

    return context, messages


def _parse_message(message: Message) -> ChatMessage:
    author = (
        ChatAuthor.BOT if message.role == Role.ASSISTANT else ChatAuthor.USER
    )
    assert message.content is not None
    return ChatMessage(
        author=author, content=collect_text_content(message.content)
    )
