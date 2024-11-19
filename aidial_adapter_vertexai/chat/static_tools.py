from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Literal, NoReturn, Self

from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from vertexai.preview.generative_models import Tool as GeminiTool

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


class DynamicRetrievalConfig(BaseModel):
    mode: Literal["MODE_DYNAMIC", "MODE_UNSPECIFIED"] | None = None
    dynamic_threshold: float | None = None

    @classmethod
    def root_validator(cls, values):
        if (
            values.get("mode") == "MODE_UNSPECIFIED"
            and values.get("dynamic_threshold") is not None
        ):
            raise ValidationError(
                "dynamic_threshold must be None when mode is MODE_UNSPECIFIED"
            )
        if values.get("dynamic_threshold") is not None:
            threshold = values.get("dynamic_threshold")
            if threshold < 0 or threshold > 1:
                raise ValidationError(
                    "dynamic_threshold must be between 0 and 1"
                )
        return values


class GoogleSearchConfig(BaseModel):
    class Config:
        extra = "forbid"

    dynamic_retrieval_config: DynamicRetrievalConfig | None = None


class GoogleSearchGroundingTool(StaticToolProcessor):
    @staticmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> List[GeminiTool] | None:
        if static_function.name == ToolName.GOOGLE_SEARCH:
            google_search_config = GoogleSearchConfig()
            if static_function.configuration:
                try:
                    google_search_config = GoogleSearchConfig.validate(
                        static_function.configuration
                    )
                except PydanticValidationError:
                    raise ValidationError(
                        "Invalid configuration for Google search tool"
                    )
            tools = [
                GeminiTool.from_dict(
                    {
                        "google_search_retrieval": google_search_config.dict(
                            exclude_none=True
                        )
                    }
                )
            ]

            return tools
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
        ret: List[GeminiTool] = []
        for tool in self.functions:
            ret.extend(
                GoogleSearchGroundingTool.parse_gemini_tools(tool)
                or unknown_tool_name(tool)
            )
        return ret

    def not_supported(self) -> None:
        if self.functions:
            raise ValidationError("Static tools aren't supported")
