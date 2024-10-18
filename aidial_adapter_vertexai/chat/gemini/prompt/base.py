from abc import ABC
from typing import List, Set

from pydantic import BaseModel
from vertexai.preview.generative_models import Content, Part

from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import Truncatable


class GeminiConversation(BaseModel):
    system_instruction: List[Part] | None = None
    contents: List[Content]

    class Config:
        arbitrary_types_allowed = True


class GeminiPrompt(BaseModel, Truncatable, ABC):
    system_instruction: List[Part] | None = None
    contents: List[Content]
    tools: ToolsConfig

    class Config:
        arbitrary_types_allowed = True

    @property
    def has_system_message(self) -> bool:
        return self.system_instruction is not None

    def keep(self, index: int) -> bool:
        # Keep the system message...
        if self.system_instruction is not None and index == 0:
            return True
        index -= self.has_system_message

        # ...and the last user message
        if index == len(self.contents) - 1:
            return True

        return False

    def __len__(self) -> int:
        return int(self.has_system_message) + len(self.contents)

    def partition(self) -> List[int]:
        n = len(self.contents)
        return [1] * self.has_system_message + [2] * (n // 2) + [1] * (n % 2)

    def select(self, indices: Set[int]) -> "GeminiPrompt":
        system_instruction: List[Part] | None = None
        contents: List[Content] = []

        offset = 0
        if self.has_system_message and 0 in indices:
            system_instruction = self.system_instruction
            offset += 1

        for idx in range(len(self.contents)):
            if idx + offset in indices:
                contents.append(self.contents[idx])

        if len(self.contents) - 1 + offset not in indices:
            raise RuntimeError("The last user prompt must not be omitted.")

        return GeminiPrompt(
            system_instruction=system_instruction,
            contents=contents,
            tools=self.tools,
        )
