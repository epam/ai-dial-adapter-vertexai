from typing import List, Literal, Self, assert_never

from aidial_sdk.chat_completion import Function, FunctionChoice, ToolChoice
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from pydantic import BaseModel
from vertexai.preview.generative_models import (
    FunctionDeclaration as GeminiFunction,
)
from vertexai.preview.generative_models import Tool as GeminiTool

from aidial_adapter_vertexai.chat.errors import ValidationError


class ToolsConfig(BaseModel):
    functions: List[Function] | None
    is_tool: bool

    def not_supported(self) -> None:
        if self.functions is not None:
            if self.is_tool:
                raise ValidationError("The tools aren't supported")
            else:
                raise ValidationError("The functions aren't supported")

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

        if request.functions is not None:

            functions = request.functions
            function_call = request.function_call
            is_tool = False

        elif request.tools is not None:
            functions = [tool.function for tool in request.tools]
            function_call = ToolsConfig.tool_choice_to_function_all(
                request.tool_choice
            )
            is_tool = True

        else:
            functions = None
            function_call = None
            is_tool = False

        selected = ToolsConfig.select_function(function_call, functions)
        return cls(functions=selected, is_tool=is_tool)

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
