from abc import ABC, abstractmethod
from enum import Enum
from typing import List, NoReturn, Self

from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)
from pydantic import BaseModel
from vertexai.preview.generative_models import Tool as GeminiTool
from vertexai.preview.generative_models import grounding

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolName(str, Enum):
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding
    GOOGLE_SEARCH = "google_search"


class StaticToolProcessor(ABC):
    @staticmethod
    @abstractmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> List[GeminiTool] | None: ...


class GoogleSearchGroundingTool(StaticToolProcessor):
    @staticmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> List[GeminiTool] | None:
        if static_function.name == ToolName.GOOGLE_SEARCH:
            if static_function.configuration:
                raise ValidationError(
                    "Google search tool doesn't support configuration"
                )
            return [
                GeminiTool.from_google_search_retrieval(
                    grounding.GoogleSearchRetrieval()
                )
            ]
        return None


def unknown_tool_name(
    static_function: StaticFunction,
) -> NoReturn:
    raise ValidationError(
        f"Unsupported static function: {static_function.name}"
    )


class StaticToolsConfig(BaseModel):
    functions: List[StaticFunction]

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self:
        if request.tools is None:
            return cls(functions=[])

        return cls(
            functions=[
                tool.static_function
                for tool in request.tools
                if isinstance(tool, StaticTool)
            ]
        )

    @classmethod
    def noop(cls) -> Self:
        return cls(functions=[])

    def to_gemini_tools(self) -> List[GeminiTool]:
        ret = []
        for tool in self.functions:
            ret.extend(
                GoogleSearchGroundingTool.parse_gemini_tools(tool)
                or unknown_tool_name(tool)
            )
        return ret

    def not_supported(self) -> None:
        if self.functions:
            raise ValidationError("Static tools aren't supported")
