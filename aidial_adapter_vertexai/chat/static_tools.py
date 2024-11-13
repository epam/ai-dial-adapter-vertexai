from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Self

from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticTool,
)
from pydantic import BaseModel
from vertexai.preview.generative_models import Tool as GeminiTool
from vertexai.preview.generative_models import grounding

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolName(str, Enum):
    """
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding
    """

    GOOGLE_SEARCH = "google_search"


class StaticToolProcessor(ABC):
    @abstractmethod
    def validate_config(self, config: dict | None) -> None: ...

    @abstractmethod
    def to_gemini_tools(self, tool: StaticTool) -> List[GeminiTool]: ...

    def process(self, tool: StaticTool) -> List[GeminiTool]:
        self.validate_config(tool.static_function.configuration)
        return self.to_gemini_tools(tool)


class GoogleSearchGroundingToolConfig(BaseModel):
    class Config:
        extra = "forbid"

    datastore: str | None = None
    project: str | None = None
    location: str | None = None
    data_store_id: str | None = None


class GoogleSearchGroundingTool(StaticToolProcessor):
    def validate_config(self, config: dict | None) -> None:
        if config:
            raise ValidationError(
                "Google search tool doesn't support configuration"
            )

    def to_gemini_tools(self, tool: StaticTool) -> List[GeminiTool]:
        return [
            GeminiTool.from_google_search_retrieval(
                grounding.GoogleSearchRetrieval()
            )
        ]


class StaticToolsConfig(BaseModel):
    tools: List[StaticTool]

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self:
        if request.tools is None:
            return cls(tools=[])

        return cls(
            tools=[
                tool for tool in request.tools if isinstance(tool, StaticTool)
            ]
        )

    @classmethod
    def noop(cls) -> Self:
        return cls(tools=[])

    def to_gemini_tools(self) -> List[GeminiTool]:
        gemini_tools = []
        for tool in self.tools:
            if tool.static_function.name == ToolName.GOOGLE_SEARCH.value:
                gemini_tools.extend(GoogleSearchGroundingTool().process(tool))
            else:
                raise ValidationError(
                    f"Unsupported static tool: {tool.static_function.name}"
                )
        return gemini_tools

    def not_supported(self) -> None:
        if len(self.tools) > 0:
            raise ValidationError("Static tools aren't supported")
