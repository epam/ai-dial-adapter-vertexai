from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Self, Set

from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaToolParam as ClaudeTool

from aidial_adapter_vertexai.chat.conversation.base import BaseConversation
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatablePrompt
from aidial_adapter_vertexai.dial_api.resource import DialResource
from aidial_adapter_vertexai.utils.list_projection import ListProjection


@dataclass
class ClaudeMessage:
    claude_message: MessageParam
    dial_resources: List[DialResource]


ClaudeConversation = BaseConversation[str | List[TextBlockParam], ClaudeMessage]


@dataclass
class ClaudePrompt(TruncatablePrompt):
    conversation: ClaudeConversation
    tools: ToolsConfig = field(default_factory=ToolsConfig.noop)

    @property
    def system(self) -> str | List[TextBlockParam] | None:
        return self.conversation.system

    @property
    def claude_messages(self) -> List[MessageParam]:
        return [m.claude_message for m in self.conversation.messages.raw_list]

    @property
    def removed_indices(self) -> List[int]:
        return list(self.conversation.messages.get_removed_indices())

    @property
    def n_messages(self) -> int:
        return len(self.conversation.messages)

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
        return int(self.has_system_instruction) + self.n_messages

    def partition_messages(self) -> List[int]:
        n = self.n_messages
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
        for idx in range(self.n_messages):
            if idx + offset in indices:
                message_indices.add(idx)

        messages: ListProjection[ClaudeMessage] = (
            self.conversation.messages.select(message_indices)
        )

        if self.n_messages - 1 + offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return self.__class__(
            conversation=ClaudeConversation(system=system, messages=messages),
            tools=self.tools,
        )

    def to_claude_tools(self) -> List[ClaudeTool] | None:
        return self.tools.to_claude_tools()

    @cached_property
    def dial_resources(self) -> List[DialResource]:
        ret: List[DialResource] = []
        for message in self.conversation.messages.raw_list:
            for resource in message.dial_resources:
                ret.append(resource)
        return ret

    def get_dial_resource(self, index: int) -> DialResource | None:
        if index < 0 or index >= len(self.dial_resources):
            return None
        return self.dial_resources[index]
