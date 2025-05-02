from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Self, Set, Tuple, Type, TypeVar

from aidial_adapter_vertexai.utils.list import MessageMergeStrategy, group_by
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

    def merge_messages_with_same_role(
        self, merger: Type[MessageMergeStrategy[MessageT]]
    ) -> Self:
        def _key(a: Tuple[MessageT, Set[int]]) -> str:
            return merger.role(a[0])

        def _merge(
            a: Tuple[MessageT, Set[int]],
            b: Tuple[MessageT, Set[int]],
        ) -> Tuple[MessageT, Set[int]]:
            (msg1, set1), (msg2, set2) = a, b
            return (merger.merge(msg1, msg2), set1 | set2)

        self.messages = ListProjection(
            self.messages.start_index,
            self.messages.end_index,
            group_by(
                lst=self.messages.list,
                key=_key,
                init=lambda x: x,
                merge=_merge,
            ),
        )

        return self
