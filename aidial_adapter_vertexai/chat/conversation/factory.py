from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from aidial_sdk.chat_completion import Message as DialMessage
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.utils.resource import Resource

PartT = TypeVar("PartT")
ContentT = TypeVar("ContentT")
ConversationT = TypeVar("ConversationT", bound=BaseConversation)


class Parts(BaseModel, Generic[PartT]):
    parts: list[PartT] = []
    resources: list[Resource] = []

    def append_part(self, part: PartT):
        self.parts.append(part)

    def append_parts(self, parts: list[PartT]):
        self.parts.extend(parts)

    def append_text_part(self, part: PartT):
        self.parts.append(part)

    def append_multi_modal_part(self, part: PartT, resource: Resource):
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
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_call_result: str,
        resources: list[Resource],
    ) -> PartT: ...

    @abstractmethod
    def create_content(
        self, idx: int, dial_message: DialMessage, parts: Parts[PartT]
    ) -> ContentT: ...

    @abstractmethod
    def create_conversation(
        self, system_instruction: list[PartT] | None, contents: list[ContentT]
    ) -> ConversationT: ...
