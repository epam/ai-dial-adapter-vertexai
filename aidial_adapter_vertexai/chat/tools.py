from enum import Enum
from typing import Dict, List, Literal, Self, assert_never

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticTool,
    Tool,
)
from anthropic.types.beta import (
    BetaToolChoiceAnyParam as ClaudeToolChoiceAnyParam,
)
from anthropic.types.beta import (
    BetaToolChoiceAutoParam as ClaudeToolChoiceAutoParam,
)
from anthropic.types.beta import (
    BetaToolChoiceNoneParam as ClaudeToolChoiceNoneParam,
)
from anthropic.types.beta import BetaToolChoiceParam as ClaudeToolChoice
from anthropic.types.beta import (
    BetaToolChoiceToolParam as ClaudeToolChoiceToolParam,
)
from anthropic.types.beta import BetaToolParam as ClaudeTool
from google.genai.types import (
    FunctionCallingConfigDict as GenAIFunctionCallingConfig,
)
from google.genai.types import (
    FunctionCallingConfigMode as GenAIFunctionCallingConfigMode,
)
from google.genai.types import (
    FunctionDeclarationDict as GenAIFunctionDeclaration,
)
from google.genai.types import ToolConfigDict as GenAIToolConfig
from google.genai.types import ToolDict as GenAITool
from pydantic.v1 import BaseModel
from vertexai.preview.generative_models import (
    FunctionDeclaration as GeminiFunction,
)
from vertexai.preview.generative_models import Tool as GeminiTool
from vertexai.preview.generative_models import ToolConfig as GeminiToolConfig

from aidial_adapter_vertexai.chat.errors import ValidationError

FunctionCallingConfig = GeminiToolConfig.FunctionCallingConfig


class ToolsMode(Enum):
    TOOLS = "TOOLS"
    FUNCTIONS = "FUNCTIONS"


_EMPTY_OBJECT_JSON_SCHEMA = {"type": "object", "properties": {}}


class ToolsConfig(BaseModel):
    tools: List[Tool]
    """
    List of functions/tools.
    """

    tool_choice: Literal["auto", "none", "required"] | ToolChoice

    tool_ids: Dict[str, str] | None
    """
    Mapping from tool call IDs to corresponding tool names.
    None means that functions are used, not tools.
    """

    @property
    def tools_mode(self) -> ToolsMode:
        if self.tool_ids is not None:
            return ToolsMode.TOOLS
        else:
            return ToolsMode.FUNCTIONS

    @property
    def is_tool(self) -> bool:
        return self.tool_ids is not None

    def not_supported(self) -> None:
        if self.tools:
            if self.is_tool:
                raise ValidationError("The tools aren't supported")
            else:
                raise ValidationError("The functions aren't supported")

    def create_fresh_tool_call_id(self, tool_name: str) -> str:
        if self.tool_ids is None:
            raise ValidationError("Function are used, but requested tool id")

        idx = 1
        while True:
            id = f"{tool_name}_{idx}"
            if id not in self.tool_ids:
                self.tool_ids[id] = tool_name
                return id
            idx += 1

    def get_tool_name(self, tool_call_id: str) -> str:
        if self.tool_ids is None:
            raise ValidationError("Function are used, but requested tool name")

        tool_name = self.tool_ids.get(tool_call_id)
        if tool_name is None:
            raise ValidationError(f"Tool call ID not found: {self.tool_ids}")
        return tool_name

    @staticmethod
    def _function_call_to_tool_choice(
        function_call: Literal["auto", "none"] | FunctionChoice | None,
    ) -> Literal["auto", "none", "required"] | ToolChoice | None:
        match function_call:
            case FunctionChoice():
                return ToolChoice(type="function", function=function_call)
            case _:
                return function_call

    @staticmethod
    def _get_tool_from_function(
        tool: Function | Tool | StaticTool,
    ) -> Tool | None:
        if isinstance(tool, StaticTool):
            # Static tools are handled separately in StaticToolsConfig
            return None
        if isinstance(tool, Function):
            return Tool(type="function", function=tool)
        else:
            return tool

    @staticmethod
    def _get_tools_from_functions(
        tools: List[Function] | List[Tool | StaticTool],
    ) -> List[Tool]:
        return [
            elem
            for tool in tools
            if (elem := ToolsConfig._get_tool_from_function(tool)) is not None
        ]

    @classmethod
    def noop(cls) -> Self:
        return cls(tools=[], tool_choice="auto", tool_ids=None)

    def is_empty(self) -> bool:
        return not self.tools

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self:
        validate_messages(request)

        if request.functions is not None:
            tools = cls._get_tools_from_functions(request.functions)
            tool_choice = cls._function_call_to_tool_choice(
                request.function_call
            )
            tool_ids = None
        elif request.tools is not None:
            tools = cls._get_tools_from_functions(request.tools)
            tool_choice = request.tool_choice
            tool_ids = collect_tool_ids(request.messages)
        else:
            return cls.noop()

        return cls(
            tools=tools,
            tool_choice=tool_choice or "auto",
            tool_ids=tool_ids,
        )

    def to_claude_tools(self) -> List[ClaudeTool] | None:
        if not self.tools:
            return None

        def _create_tool(tool: Tool) -> ClaudeTool:
            func = tool.function
            ret: ClaudeTool = {
                "name": func.name,
                "input_schema": func.parameters or _EMPTY_OBJECT_JSON_SCHEMA,
            }
            if func.description:
                ret["description"] = func.description
            return ret

        return [_create_tool(tool) for tool in self.tools]

    def to_claude_tool_choice(self) -> ClaudeToolChoice | None:
        if not self.tools:
            return None

        match self.tool_choice:
            case "auto":
                return ClaudeToolChoiceAutoParam(type="auto")
            case "none":
                return ClaudeToolChoiceNoneParam(type="none")
            case "required":
                return ClaudeToolChoiceAnyParam(type="any")
            case ToolChoice(function=function):
                return ClaudeToolChoiceToolParam(
                    type="tool", name=function.name
                )
            case _:
                assert_never(self.tool_choice)

    def to_gemini_tools(self) -> List[GeminiTool]:
        if not self.tools:
            return []

        return [
            GeminiTool(
                function_declarations=[
                    GeminiFunction(
                        name=tool.function.name,
                        parameters=tool.function.parameters
                        or _EMPTY_OBJECT_JSON_SCHEMA,
                        description=tool.function.description,
                    )
                    for tool in self.tools
                ]
            )
        ]

    def to_gemini_tool_config(self) -> GeminiToolConfig | None:
        if not self.tools:
            return None

        if self.tool_choice == "required":
            return GeminiToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfig.Mode.ANY,
                    allowed_function_names=[
                        tool.function.name for tool in self.tools
                    ],
                )
            )
        else:
            return GeminiToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfig.Mode.AUTO
                )
            )

    def to_gemini_genai_tools(self) -> List[GenAITool]:
        if not self.tools:
            return []

        return [
            GenAITool(
                function_declarations=[
                    GenAIFunctionDeclaration(
                        name=tool.function.name,
                        parameters_json_schema=tool.function.parameters
                        or _EMPTY_OBJECT_JSON_SCHEMA,
                        description=tool.function.description,
                    )
                    for tool in self.tools
                ]
            )
        ]

    def to_gemini_genai_tool_config(self) -> GenAIToolConfig | None:
        if not self.tools:
            return None

        match self.tool_choice:
            case "auto":
                mode, names = GenAIFunctionCallingConfigMode.AUTO, None
            case "none":
                mode, names = GenAIFunctionCallingConfigMode.NONE, None
            case "required":
                mode, names = GenAIFunctionCallingConfigMode.ANY, None
            case ToolChoice(function=function):
                mode, names = GenAIFunctionCallingConfigMode.ANY, [
                    function.name
                ]

        return GenAIToolConfig(
            function_calling_config=GenAIFunctionCallingConfig(
                mode=mode, allowed_function_names=names
            )
        )


def validate_messages(request: AzureChatCompletionRequest) -> None:
    decl_tools = request.tools is not None
    decl_functions = request.functions is not None

    if decl_functions and decl_tools:
        raise ValidationError("Both functions and tools are not allowed")

    for message in request.messages:
        if message.role == Role.ASSISTANT:
            use_tools = message.tool_calls is not None
            if use_tools and not decl_tools:
                raise ValidationError(
                    "Assistant message uses tools, but tools are not declared"
                )

            use_functions = message.function_call is not None
            if use_functions and not decl_functions:
                raise ValidationError(
                    "Assistant message uses functions, but functions are not declared"
                )
        if message.role == Role.FUNCTION:
            if not decl_functions:
                raise ValidationError(
                    "Function message is used, but functions are not declared"
                )
        if message.role == Role.TOOL:
            if not decl_tools:
                raise ValidationError(
                    "Tool message is used, but tools are not declared"
                )


def collect_tool_ids(messages: List[Message]) -> Dict[str, str]:
    ret: Dict[str, str] = {}

    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                ret[tool_call.id] = tool_call.function.name

    return ret
