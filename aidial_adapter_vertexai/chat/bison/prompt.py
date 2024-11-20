from enum import Enum
from typing import List, Optional, Set, Tuple

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.language_models import ChatMessage, ChatSession

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatablePrompt
from aidial_adapter_vertexai.dial_api.request import collect_text_content


class ChatAuthor(str, Enum):
    USER = ChatSession.USER_AUTHOR
    BOT = ChatSession.MODEL_AUTHOR

    def __repr__(self) -> str:
        return f"{self.value!r}"


class BisonPrompt(BaseModel, TruncatablePrompt):
    system_instruction: Optional[str] = None
    history: List[ChatMessage] = []
    last_user_message: str

    @classmethod
    def parse(cls, history: List[Message]) -> "BisonPrompt":
        system_instruction, history, last_user_message = (
            _validate_and_split_messages(history)
        )

        return cls(
            system_instruction=system_instruction,
            history=list(map(_to_bison_message, history)),
            last_user_message=last_user_message,
        )

    @property
    def has_system_instruction(self) -> bool:
        return self.system_instruction is not None

    def is_required_message(self, index: int) -> bool:
        # Keep the system instruction...
        if self.has_system_instruction and index == 0:
            return True

        # ...and the last user message
        if index == len(self) - 1:
            return True

        return False

    def __len__(self) -> int:
        return int(self.has_system_instruction) + len(self.history) + 1

    def partition_messages(self) -> List[int]:
        n = len(self.history)
        return (
            [1] * self.has_system_instruction
            + [2] * (n // 2)
            + [1] * (n % 2)
            + [1]
        )

    def select(self, indices: Set[int]) -> "BisonPrompt":
        system_instruction: str | None = None
        history: List[ChatMessage] = []

        offset = 0
        if self.has_system_instruction and 0 in indices:
            system_instruction = self.system_instruction
            offset += 1

        for idx in range(len(self.history)):
            if idx + offset in indices:
                history.append(self.history[idx])
        offset += len(self.history)

        if offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return BisonPrompt(
            system_instruction=system_instruction,
            history=history,
            last_user_message=self.last_user_message,
        )


_SUPPORTED_ROLES = {Role.SYSTEM, Role.USER, Role.ASSISTANT}


def _validate_and_split_messages(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message], str]:
    if len(messages) == 0:
        raise ValidationError("The chat history must have at least one message")

    for message in messages:
        if message.content is None:
            raise ValidationError("Message content must be present")

        if message.role not in _SUPPORTED_ROLES:
            raise ValidationError(
                f"Message role must be one of {_SUPPORTED_ROLES}"
            )

    if len(messages) > 0 and messages[0].role == Role.SYSTEM:
        system_message, *history = messages
        system_instruction = collect_text_content(system_message.content)
        system_instruction = (
            system_instruction if system_instruction.strip() else None
        )
    else:
        system_instruction, history = None, messages

    if len(history) == 0 and system_instruction is not None:
        raise ValidationError(
            "The chat history must have at least one non-system message"
        )

    role: Optional[Role] = None
    for message in history:
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

    if len(history) % 2 == 0:
        raise ValidationError(
            "There should be odd number of messages for correct alternating turn"
        )

    *history, last_message = history

    if last_message.role != Role.USER:
        raise ValidationError("The last message must be a user message")

    return (
        system_instruction,
        history,
        collect_text_content(last_message.content),
    )


def _to_bison_message(message: Message) -> ChatMessage:
    author = (
        ChatAuthor.BOT if message.role == Role.ASSISTANT else ChatAuthor.USER
    )
    return ChatMessage(
        author=author, content=collect_text_content(message.content)
    )
