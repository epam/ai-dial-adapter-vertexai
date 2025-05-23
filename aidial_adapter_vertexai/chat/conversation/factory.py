from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from aidial_sdk.chat_completion import Message as DialMessage

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation

PartT = TypeVar("PartT")
ContentT = TypeVar("ContentT")
ConversationT = TypeVar("ConversationT", bound=BaseConversation)


class ConversationFactoryBase(ABC, Generic[PartT, ContentT, ConversationT]):
    @abstractmethod
    def create_multi_modal_part(self, data: bytes, mime_type: str) -> PartT: ...

    @abstractmethod
    def create_text_part(self, text: str) -> PartT: ...

    @abstractmethod
    def create_function_call_part(
        self, name: str, args: str, tool_call_id: str
    ) -> PartT: ...

    @abstractmethod
    def create_function_result_part(
        self, name: str, args: str, tool_call_id: str
    ) -> PartT: ...

    @abstractmethod
    def create_content(
        self, dial_message: DialMessage, parts: List[PartT]
    ) -> ContentT: ...

    @abstractmethod
    def create_conversation(
        self, system_instruction: List[PartT] | None, contents: List[ContentT]
    ) -> ConversationT: ...
