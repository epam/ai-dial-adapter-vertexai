from enum import Enum
from typing import List, Literal, NoReturn, Self

from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)
from google.genai.types import GoogleSearchDict as GenAIGoogleSearch
from google.genai.types import ToolDict as GenAITool
from pydantic.v1 import BaseModel, ConstrainedFloat, Field
from pydantic.v1 import ValidationError as PydanticValidationError
from pydantic.v1 import root_validator
from vertexai.preview.generative_models import Tool as GeminiTool

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolName(str, Enum):
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding
    # https://ai.google.dev/gemini-api/docs/grounding?lang=python#google-search-retrieval
    GOOGLE_SEARCH = "google_search"


class DynamicThreshold(ConstrainedFloat):
    ge = 0
    le = 1


class DynamicRetrievalConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_population_by_field_name = True

    mode: Literal["MODE_DYNAMIC", "MODE_UNSPECIFIED"] | None = None
    dynamic_threshold: DynamicThreshold | None = Field(
        None, alias="dynamicThreshold"
    )

    @root_validator(pre=True)
    def check_dynamic_threshold(cls, values):
        if values.get("mode") == "MODE_UNSPECIFIED" and (
            values.get("dynamic_threshold") is not None
            or values.get("dynamicThreshold") is not None
        ):
            raise ValidationError(
                "dynamic_threshold must be None when mode is MODE_UNSPECIFIED"
            )
        return values


class GoogleSearchConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_population_by_field_name = True

    dynamic_retrieval_config: DynamicRetrievalConfig | None = Field(
        None, alias="dynamicRetrievalConfig"
    )


class GoogleSearchGroundingTool:
    @staticmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> List[GeminiTool] | None:
        if static_function.name == ToolName.GOOGLE_SEARCH:
            google_search_config = GoogleSearchConfig(
                dynamicRetrievalConfig=None
            )
            if static_function.configuration:
                try:
                    google_search_config = GoogleSearchConfig.validate(
                        static_function.configuration
                    )
                except PydanticValidationError:
                    raise ValidationError(
                        "Invalid configuration for Google search tool"
                    )
            return [
                GeminiTool.from_dict(
                    {
                        "google_search_retrieval": google_search_config.dict(
                            exclude_none=True
                        )
                    }
                )
            ]

        return None


class GenAIGoogleSearchTool:
    @staticmethod
    def parse_gemini_tools(
        is_gemini_1_5: bool,
        static_function: StaticFunction,
    ) -> List[GenAITool] | None:
        if static_function.name == ToolName.GOOGLE_SEARCH:
            config = static_function.configuration
            if is_gemini_1_5:
                return [GenAITool(google_search_retrieval=config)]  # type: ignore
            else:
                if config:
                    raise ValidationError(
                        "Model doesn't support configuration for Google search tool"
                    )
                return [GenAITool(google_search=GenAIGoogleSearch())]
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

    def to_gemini_genai_tools(self, is_gemini_1_5: bool) -> List[GenAITool]:
        ret: List[GenAITool] = []
        for tool in self.functions:
            ret.extend(
                GenAIGoogleSearchTool.parse_gemini_tools(is_gemini_1_5, tool)
                or unknown_tool_name(tool)
            )
        return ret

    def not_supported(self) -> None:
        if self.functions:
            raise ValidationError("Static tools aren't supported")

    def is_empty(self) -> bool:
        return not self.functions
