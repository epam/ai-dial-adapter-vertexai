from typing import Dict, List, Literal, Self, assert_never

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from pydantic import BaseModel
from vertexai.preview.generative_models import (
    FunctionDeclaration as GeminiFunction,
)
from vertexai.preview.generative_models import Tool as GeminiTool

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolsConfig(BaseModel):
    functions: List[Function] | None
    """
    List of functions/tools
    """

    tool_ids: Dict[str, str] | None
    """
    Mapping from tool call IDs to corresponding tool names.
    If None, then function are used, not tools.
    """

    @property
    def is_tool(self) -> bool:
        return self.tool_ids is not None

    def not_supported(self) -> None:
        if self.functions is not None:
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
    def select_function(
        function_call: Literal["auto", "none"] | FunctionChoice | None,
        functions: List[Function] | None,
    ) -> List[Function] | None:
        if functions is None:
            return None

        match function_call:
            case None:
                return None
            case "none":
                return None
            case "auto":
                return functions
            case FunctionChoice(name=name):
                # NOTE: there is way to configure ToolsConfig, but it's not
                # possible to pass to ChatSession.
                new_functions = [
                    func for func in functions if func.name == name
                ]
                return None if len(new_functions) == 0 else new_functions
            case _:
                assert_never(function_call)

    @staticmethod
    def tool_choice_to_function_all(
        tool_choice: Literal["auto", "none"] | ToolChoice | None,
    ) -> Literal["auto", "none"] | FunctionChoice | None:
        match tool_choice:
            case None:
                return None
            case "none":
                return "none"
            case "auto":
                return "auto"
            case ToolChoice(function=FunctionChoice(name=name)):
                return FunctionChoice(name=name)
            case _:
                assert_never(tool_choice)

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self:
        validate_messages(request)

        if request.functions is not None:
            functions = request.functions
            function_call = request.function_call
            tool_ids = None

        elif request.tools is not None:
            functions = [tool.function for tool in request.tools]
            function_call = ToolsConfig.tool_choice_to_function_all(
                request.tool_choice
            )
            tool_ids = collect_tool_ids(request.messages)

        else:
            functions = None
            function_call = None
            tool_ids = None

        selected = ToolsConfig.select_function(function_call, functions)
        return cls(functions=selected, tool_ids=tool_ids)

    def to_gemini_tools(self) -> List[GeminiTool] | None:
        if self.functions is None:
            return None

        return [
            GeminiTool(
                function_declarations=[
                    GeminiFunction(
                        name=func.name,
                        parameters=func.parameters,
                        description=func.description,
                    )
                ]
            )
            for func in self.functions
        ]


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
