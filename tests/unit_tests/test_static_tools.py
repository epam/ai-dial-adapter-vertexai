import pytest
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    Function,
    StaticFunction,
    StaticTool,
    Tool,
)

from aidial_adapter_vertexai.chat.gemini.generation_config import (
    create_genai_count_tokens_config,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig


def test_no_tools():
    request = AzureChatCompletionRequest(messages=[])
    config = StaticToolsConfig.from_request(request)
    assert config.functions == []


@pytest.mark.parametrize(
    ("function_names", "expected_count"),
    [
        (["weather"], 1),
        (["weather", "news"], 2),
    ],
)
def test_merge_function_tools_and_static_tool_for_gemini_config(
    function_names: list[str], expected_count: int
):
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
                static_function=StaticFunction(name="google_search"),
            ),
        ],
    )

    tools = ToolsConfig.from_request(request)
    static_tools = StaticToolsConfig.from_request(request)
    config = create_genai_count_tokens_config(tools, static_tools)

    toolsets = config.get("tools")
    assert toolsets is not None
    assert len(toolsets) == 1

    merged_toolset = toolsets[0]
    assert "google_search" in merged_toolset
    function_declarations = merged_toolset.get("function_declarations")
    assert function_declarations is not None
    assert len(function_declarations) == expected_count
    assert {decl.get("name") for decl in function_declarations} == set(
        function_names
    )
