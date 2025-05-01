from enum import Enum
from typing import Dict, List, Literal, Self, Tuple, assert_never, cast

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest, Tool
from anthropic.types import ToolChoiceParam as ClaudeToolConfig
from anthropic.types import ToolParam as ClaudeTool
from google.genai.types import (
    FunctionCallingConfigDict as GenAIFunctionCallingConfig,
)
from google.genai.types import (
    FunctionCallingConfigMode as GenAIFunctionCallingConfigMode,
)
from google.genai.types import (
    FunctionDeclarationDict as GenAIFunctionDeclaration,
)
from google.genai.types import SchemaDict as GenAISchema
from google.genai.types import ToolConfigDict as GenAIToolConfig
from google.genai.types import ToolDict as GenAITool
from google.genai.types import Type as GenAIType
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


class ToolsConfig(BaseModel):
    functions: List[Function]
    """
    List of functions/tools.
    """

    required: bool
    """
    True forces the model to call one of the available functions.
    False allows the model to pick between generating a message or
    calling one or more tools/functions.
    """

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
        if self.functions:
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
    def filter_functions(
        function_call: Literal["auto", "none", "required"] | FunctionChoice,
        functions: List[Function],
    ) -> Tuple[bool, List[Function]]:
        match function_call:
            case "none":
                return False, []
            case "auto":
                return False, functions
            case "required":
                return True, functions
            case FunctionChoice(name=name):
                new_functions = [
                    func for func in functions if func.name == name
                ]
                if not new_functions:
                    raise ValidationError(
                        f"Function {name!r} is not on the list of available functions"
                    )
                return True, new_functions
            case _:
                assert_never(function_call)

    @staticmethod
    def tool_choice_to_function_call(
        tool_choice: Literal["auto", "none", "required"] | ToolChoice | None,
    ) -> Literal["auto", "none", "required"] | FunctionChoice | None:
        match tool_choice:
            case ToolChoice(function=FunctionChoice(name=name)):
                return FunctionChoice(name=name)
            case _:
                return tool_choice

    @classmethod
    def noop(cls) -> Self:
        return cls(functions=[], required=False, tool_ids=None)

    def is_empty(self) -> bool:
        return not self.functions

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self:
        validate_messages(request)

        if request.functions is not None:
            functions = request.functions
            function_call = request.function_call
            tool_ids = None

        elif request.tools is not None:
            functions = [
                tool.function
                for tool in request.tools
                if isinstance(tool, Tool)
            ]
            function_call = ToolsConfig.tool_choice_to_function_call(
                request.tool_choice
            )
            tool_ids = collect_tool_ids(request.messages)

        else:
            functions = []
            function_call = None
            tool_ids = None

        if function_call is None:
            function_call = "auto" if functions else "none"

        required, selected = ToolsConfig.filter_functions(
            function_call, functions
        )

        return cls(functions=selected, required=required, tool_ids=tool_ids)

    def to_claude_tools(self) -> List[ClaudeTool] | None:
        if not self.functions:
            return None

        def _create_tool(func: Function) -> ClaudeTool:
            tool: ClaudeTool = {
                "name": func.name,
                "input_schema": func.parameters
                or {"type": "object", "properties": {}},
            }
            if func.description:
                tool["description"] = func.description
            return tool

        return [_create_tool(func) for func in self.functions]

    def to_claude_tool_config(self) -> ClaudeToolConfig | None:
        if not self.functions:
            return None

        if self.required:
            return {"type": "any"}
        else:
            return {"type": "auto"}

    def to_gemini_tools(self) -> List[GeminiTool]:
        if not self.functions:
            return []

        return [
            GeminiTool(
                function_declarations=[
                    GeminiFunction(
                        name=func.name,
                        parameters=func.parameters
                        or {"type": "object", "properties": {}},
                        description=func.description,
                    )
                    for func in self.functions
                ]
            )
        ]

    def to_gemini_tool_config(self) -> GeminiToolConfig | None:
        if not self.functions:
            return None

        if self.required:
            return GeminiToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfig.Mode.ANY,
                    allowed_function_names=[
                        func.name for func in self.functions
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
        if not self.functions:
            return []

        return [
            GenAITool(
                function_declarations=[
                    GenAIFunctionDeclaration(
                        name=func.name,
                        parameters=(
                            _convert_genai_function_parameters(func.parameters)
                            if func.parameters
                            else GenAISchema(
                                type=GenAIType.OBJECT, properties={}
                            )
                        ),
                        description=func.description,
                    )
                    for func in self.functions
                ]
            )
        ]

    def to_gemini_genai_tool_config(self) -> GenAIToolConfig | None:
        if not self.functions:
            return None

        if self.required:
            return GenAIToolConfig(
                function_calling_config=GenAIFunctionCallingConfig(
                    mode=GenAIFunctionCallingConfigMode.ANY,
                    allowed_function_names=[
                        func.name for func in self.functions
                    ],
                )
            )
        else:
            return GenAIToolConfig(
                function_calling_config=GenAIFunctionCallingConfig(
                    mode=GenAIFunctionCallingConfigMode.AUTO,
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


_JSON_SCHEMA_TYPES = [
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
]


def _convert_genai_function_parameters(function_schema: dict) -> GenAISchema:
    def _convert(value):
        match value:
            case dict():
                d = {k: _convert(v) for k, v in value.items()}

                # GenAI lib requires property types to be in uppercase
                if (ty := d.get("type")) in _JSON_SCHEMA_TYPES:
                    d["type"] = ty.upper()
                return d
            case list():
                return [_convert(item) for item in value]
            case _:
                return value

    return cast(GenAISchema, _convert(function_schema))
