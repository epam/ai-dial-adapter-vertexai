from abc import ABC
from typing import List, Optional

from pydantic import BaseModel
from vertexai.preview.generative_models import Content, Part

from aidial_adapter_vertexai.chat.tools import ToolsConfig


class GeminiConversation(BaseModel):
    system_instruction: Optional[List[Part]] = None
    contents: List[Content]

    class Config:
        arbitrary_types_allowed = True


class GeminiPrompt(BaseModel, ABC):
    conversation: GeminiConversation
    tools: ToolsConfig
