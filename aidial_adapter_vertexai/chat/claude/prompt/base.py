from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Self, Set

from anthropic.types import MessageParam, TextBlockParam
from anthropic.types import ToolParam as ClaudeTool

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatablePrompt
from aidial_adapter_vertexai.utils.list_projection import ListProjection

ClaudeConversation = BaseConversation[str | List[TextBlockParam], MessageParam]


@dataclass
class ClaudePrompt(TruncatablePrompt):
    conversation: ClaudeConversation
    tools: ToolsConfig = field(default_factory=ToolsConfig.noop)

    @property
    def system(self) -> str | List[TextBlockParam] | None:
        return self.conversation.system

    @property
    def messages(self) -> ListProjection[MessageParam]:
        return self.conversation.messages

    @property
    def has_system_instruction(self) -> bool:
        return self.system is not None

    def is_required_message(self, index: int) -> bool:
        # Keep the system message...
        if self.has_system_instruction and index == 0:
            return True

        # ...and the last user message
        if index == len(self) - 1:
            return True

        return False

    def __len__(self) -> int:
        return int(self.has_system_instruction) + len(self.messages)

    def partition_messages(self) -> List[int]:
        n = len(self.messages)
        return (
            [1] * self.has_system_instruction + [2] * (n // 2) + [1] * (n % 2)
        )

    def select(self, indices: Set[int]) -> Self:

        if self.has_system_instruction and 0 in indices:
            system = self.system
        else:
            system = None

        offset = int(self.has_system_instruction)

        message_indices: Set[int] = set()
        for idx in range(len(self.messages)):
            if idx + offset in indices:
                message_indices.add(idx)

        messages: ListProjection[MessageParam] = (
            self.conversation.messages.select(message_indices)
        )

        if len(self.messages) - 1 + offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return self.__class__(
            conversation=ClaudeConversation(system=system, messages=messages),
            tools=self.tools,
        )

    def to_claude_tools(self) -> List[ClaudeTool] | None:
        return self.tools.to_claude_tools()
