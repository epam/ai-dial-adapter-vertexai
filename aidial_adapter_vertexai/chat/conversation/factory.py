from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from aidial_sdk.chat_completion import Message as DialMessage
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.dial_api.resource import DialResource

PartT = TypeVar("PartT")
ContentT = TypeVar("ContentT")
ConversationT = TypeVar("ConversationT", bound=BaseConversation)


class Parts(BaseModel, Generic[PartT]):
    parts: List[PartT] = []
    resources: List[DialResource] = []

    def append_text_part(self, part: PartT):
        self.parts.append(part)

    def append_multi_modal_part(self, part: PartT, resource: DialResource):
        self.resources.append(resource)
        self.parts.append(part)

    def empty(self) -> bool:
        return len(self.parts) == 0

    def has_text_parts(self) -> bool:
        return len(self.parts) > len(self.resources)


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
        self, idx: int, dial_message: DialMessage, parts: Parts[PartT]
    ) -> ContentT: ...

    @abstractmethod
    def create_conversation(
        self, system_instruction: List[PartT] | None, contents: List[ContentT]
    ) -> ConversationT: ...
