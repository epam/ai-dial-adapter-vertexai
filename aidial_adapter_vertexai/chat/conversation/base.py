from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Self, TypeVar

from aidial_adapter_vertexai.utils.list_projection import ListProjection

SystemT = TypeVar("SystemT")
MessageT = TypeVar("MessageT")


# NOTE: it's not pydantic BaseModel, because
# Claude's MessageParam.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class BaseConversation(Generic[SystemT, MessageT]):
    system: SystemT | None
    messages: ListProjection[MessageT]

    @classmethod
    def create(
        cls, *, system: SystemT | None = None, messages: List[MessageT]
    ) -> Self:
        return cls(
            system=system,
            messages=ListProjection.create(
                messages, idx_offset=int(system is not None)
            ),
        )

    def on_messages(
        self,
        func: Callable[[ListProjection[MessageT]], ListProjection[MessageT]],
    ) -> BaseConversation[SystemT, MessageT]:
        return BaseConversation(
            system=self.system, messages=func(self.messages)
        )
