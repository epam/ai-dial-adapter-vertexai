from dataclasses import dataclass, field
from typing import List, Self, Set

from anthropic.types import MessageParam, TextBlockParam
from anthropic.types import ToolParam as ClaudeTool

from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatablePrompt


# NOTE: it's not pydantic BaseModel, because
# MessageParam.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class ClaudeConversation:
    system: str | List[TextBlockParam] | None
    messages: List[MessageParam]


@dataclass
class ClaudePrompt(TruncatablePrompt):
    conversation: ClaudeConversation
    tools: ToolsConfig = field(default_factory=ToolsConfig.noop)

    @property
    def system(self) -> str | List[TextBlockParam] | None:
        return self.conversation.system

    @property
    def messages(self) -> List[MessageParam]:
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
        system: str | List[TextBlockParam] | None = None
        messages: List[MessageParam] = []

        offset = 0
        if self.has_system_instruction and 0 in indices:
            system = self.system
            offset += 1

        for idx in range(len(self.messages)):
            if idx + offset in indices:
                messages.append(self.messages[idx])

        if len(self.messages) - 1 + offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return self.__class__(
            conversation=ClaudeConversation(system=system, messages=messages),
            tools=self.tools,
        )

    def to_claude_tools(self) -> List[ClaudeTool] | None:
        return self.tools.to_claude_tools()
