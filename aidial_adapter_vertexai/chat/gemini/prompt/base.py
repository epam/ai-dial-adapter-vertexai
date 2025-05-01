from typing import Generic, List, Self, Set, TypeVar

from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from pydantic.v1 import BaseModel, Field
from vertexai.preview.generative_models import Content, Part
from vertexai.preview.generative_models import Tool as GeminiTool

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatablePrompt

SystemT = TypeVar("SystemT")
MessageT = TypeVar("MessageT")


class GeminiBasePrompt(
    BaseModel, TruncatablePrompt, Generic[SystemT, MessageT]
):
    system_instruction: SystemT | None = None
    messages: List[MessageT]

    tools: ToolsConfig = Field(default_factory=ToolsConfig.noop)
    static_tools: StaticToolsConfig = Field(
        default_factory=StaticToolsConfig.noop
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def has_system_instruction(self) -> bool:
        return self.system_instruction is not None

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
        system_instruction: SystemT | None = None
        messages: List[MessageT] = []

        offset = 0
        if self.has_system_instruction and 0 in indices:
            system_instruction = self.system_instruction
            offset += 1

        for idx in range(len(self.messages)):
            if idx + offset in indices:
                messages.append(self.messages[idx])

        if len(self.messages) - 1 + offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return self.__class__(
            system_instruction=system_instruction,
            messages=messages,
            tools=self.tools,
            static_tools=self.static_tools,
        )

    def to_gemini_tools(self) -> List[GeminiTool]:
        regular_tools = self.tools.to_gemini_tools()
        static_tools = self.static_tools.to_gemini_tools()
        return regular_tools + static_tools


GeminiPrompt = GeminiBasePrompt[List[Part], Content]
GeminiGenAIPrompt = GeminiBasePrompt[List[GenAIPart], GenAIContent]
