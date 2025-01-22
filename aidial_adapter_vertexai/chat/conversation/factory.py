from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from aidial_sdk.chat_completion.request import Role

PartT = TypeVar("PartT")
ContentT = TypeVar("ContentT")
ConversationT = TypeVar("ConversationT")


class ConversationFactoryBase(ABC, Generic[PartT, ContentT, ConversationT]):
    @abstractmethod
    def create_multi_modal_part(self, data: bytes, mime_type: str) -> PartT: ...

    @abstractmethod
    def create_text_part(self, text: str) -> PartT: ...

    @abstractmethod
    def create_function_call_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> PartT: ...

    @abstractmethod
    def create_function_result_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> PartT: ...

    @abstractmethod
    def create_content(self, role: Role, parts: List[PartT]) -> ContentT: ...

    @abstractmethod
    def create_conversation(
        self, system_instruction: List[PartT] | None, contents: List[ContentT]
    ) -> ConversationT: ...
