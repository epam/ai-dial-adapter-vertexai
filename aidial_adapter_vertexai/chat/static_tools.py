from enum import Enum
from typing import NoReturn, Self

from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)
from google.genai.types import GoogleSearchDict as GenAIGoogleSearch
from google.genai.types import ToolCodeExecutionDict as GenAICodeExecution
from google.genai.types import ToolDict as GenAITool
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolName(str, Enum):
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding
    # https://ai.google.dev/gemini-api/docs/grounding?lang=python#google-search-retrieval
    GOOGLE_SEARCH = "google_search"
    # https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/code-execution-api
    # https://ai.google.dev/gemini-api/docs/code-execution?lang=python
    CODE_EXECUTION = "code_execution"
    # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
    WEB_SEARCH = "web_search"


class GenAIGoogleSearchTool:
    @staticmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> list[GenAITool] | None:
        if static_function.name == ToolName.GOOGLE_SEARCH:
            if static_function.configuration:
                raise ValidationError("Google search tool isn't configurable")
            return [GenAITool(google_search=GenAIGoogleSearch())]
        return None


class GenAICodeExecutionTool:
    @staticmethod
    def parse_gemini_tools(
        static_function: StaticFunction,
    ) -> list[GenAITool] | None:
        if static_function.name == ToolName.CODE_EXECUTION:
            if static_function.configuration:
                raise ValidationError("Code execution tool isn't configurable")
            return [GenAITool(code_execution=GenAICodeExecution())]
        return None


def unknown_tool_name(static_function: StaticFunction) -> NoReturn:
    raise ValidationError(
        f"Unsupported static function: {static_function.name}"
    )


class StaticToolsConfig(BaseModel):
    functions: list[StaticFunction]

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

    def to_gemini_genai_tools(self) -> list[GenAITool]:
        ret: list[GenAITool] = []
        for tool in self.functions:
            ret.extend(
                GenAIGoogleSearchTool.parse_gemini_tools(tool)
                or GenAICodeExecutionTool.parse_gemini_tools(tool)
                or unknown_tool_name(tool)
            )
        return ret

    def not_supported(self) -> None:
        if self.functions:
            raise ValidationError("Static tools aren't supported")

    def is_empty(self) -> bool:
        return not self.functions
