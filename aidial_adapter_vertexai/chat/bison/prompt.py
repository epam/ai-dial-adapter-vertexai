from enum import Enum
from typing import List, Optional, Set, Tuple

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.language_models import ChatMessage, ChatSession

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.truncate_prompt import Truncatable
from aidial_adapter_vertexai.dial_api.request import collect_text_content


class ChatAuthor(str, Enum):
    USER = ChatSession.USER_AUTHOR
    BOT = ChatSession.MODEL_AUTHOR

    def __repr__(self) -> str:
        return f"{self.value!r}"


class BisonPrompt(BaseModel, Truncatable):
    context: Optional[str] = None
    history: List[ChatMessage] = []
    prompt: str

    @classmethod
    def parse(cls, messages: List[Message]) -> "BisonPrompt":
        context, messages = _validate_and_split_messages(messages)
        bison_messages = list(map(_parse_message, messages))

        return cls(
            context=context,
            history=bison_messages[:-1],
            prompt=bison_messages[-1].content,
        )

    @property
    def has_system_message(self) -> bool:
        return self.context is not None

    def keep(self, index: int) -> bool:
        # Keep the system message...
        if self.context is not None and index == 0:
            return True
        index -= self.has_system_message

        # ...and the last user message
        if index == len(self.history):
            return True

        return False

    def __len__(self) -> int:
        return int(self.has_system_message) + len(self.history) + 1

    def partition(self) -> List[int]:
        n = len(self.history)
        return (
            [1] * self.has_system_message + [2] * (n // 2) + [1] * (n % 2) + [1]
        )

    def select(self, indices: Set[int]) -> "BisonPrompt":
        context: str | None = None
        history: List[ChatMessage] = []

        offset = 0
        if self.has_system_message and 0 in indices:
            context = self.context
            offset += 1

        for idx in range(len(self.history)):
            if idx + offset in indices:
                history.append(self.history[idx])
        offset += len(self.history)

        if offset not in indices:
            print(self)
            print(indices)
            raise RuntimeError("The last user prompt must not be omitted.")

        return BisonPrompt(
            context=context,
            history=history,
            prompt=self.prompt,
        )


_SUPPORTED_ROLES = {Role.SYSTEM, Role.USER, Role.ASSISTANT}


def _validate_and_split_messages(
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
