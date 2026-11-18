import pytest
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    Function,
    StaticFunction,
    StaticTool,
    Tool,
)
from google.genai._transformers import t_tools
from google.genai.types import ToolDict as GenAITool

from aidial_adapter_vertexai.chat.gemini.generation_config import (
    create_genai_count_tokens_config,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig


def test_no_tools():
    request = AzureChatCompletionRequest(messages=[])
    config = StaticToolsConfig.from_request(request)
    assert config.functions == []


def _create_tools_config(
    function_names: list[str], static_tool_name: str
) -> list[GenAITool]:
    request = AzureChatCompletionRequest(
        messages=[],
        tools=[
            *[
                Tool(
                    type="function",
                    function=Function(name=name),
                )
                for name in function_names
            ],
            StaticTool(
                type="static_function",
                static_function=StaticFunction(name=static_tool_name),
            ),
        ],
    )

    tools = ToolsConfig.from_request(request)
    static_tools = StaticToolsConfig.from_request(request)
    config = create_genai_count_tokens_config(tools, static_tools)

    toolsets = config.get("tools")
    assert toolsets is not None
    return toolsets


@pytest.mark.parametrize(
    "function_names",
    [["weather"], ["weather", "news"]],
)
def test_separate_function_tools_and_static_tool_for_gemini_config(
    function_names: list[str],
):
    toolsets = _create_tools_config(function_names, "google_search")
    assert len(toolsets) == 2

    function_toolset, static_toolset = toolsets
    assert static_toolset == {"google_search": {}}
    function_declarations = function_toolset.get("function_declarations")
    assert function_declarations is not None
    assert [decl.get("name") for decl in function_declarations] == (
        function_names
    )


@pytest.mark.parametrize(
    "static_tool_name", ["google_search", "code_execution"]
)
def test_static_tool_survives_genai_tools_normalization(
    static_tool_name: str,
):
    toolsets = _create_tools_config(["weather"], static_tool_name)

    normalized = t_tools(None, toolsets)  # type: ignore
    assert len(normalized) == 2
    assert any(
        getattr(tool, static_tool_name) is not None for tool in normalized
    )
    assert any(
        [decl.name for decl in tool.function_declarations or []] == ["weather"]
        for tool in normalized
    )
