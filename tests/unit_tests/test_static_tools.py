import pytest
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.static_tools import (
    StaticToolsConfig,
    ToolName,
)


def test_no_tools():
    request = AzureChatCompletionRequest(messages=[])
    config = StaticToolsConfig.from_request(request)
    assert config.functions == []


def test_normal_google_search():
    tool = StaticTool(
        type="static_function",
        static_function=StaticFunction(
            name=ToolName.GOOGLE_SEARCH.value, configuration={}
        ),
    )
    request = AzureChatCompletionRequest(messages=[], tools=[tool])
    config = StaticToolsConfig.from_request(request)

    gemini_tools = config.to_gemini_tools()
    assert len(gemini_tools) == 1


def test_invalid_google_search_config():
    tool = StaticTool(
        type="static_function",
        static_function=StaticFunction(
            name=ToolName.GOOGLE_SEARCH.value,
            configuration={"invalid_field": "value"},  # Invalid config
        ),
    )
    request = AzureChatCompletionRequest(messages=[], tools=[tool])
    config = StaticToolsConfig.from_request(request)

    with pytest.raises(ValidationError) as exc_info:
        config.to_gemini_tools()

    assert "Google search tool doesn't support configuration" in str(
        exc_info.value
    )


def test_unknown_tool():
    tool = StaticTool(
        type="static_function",
        static_function=StaticFunction(
            name="unknown_function", configuration={}
        ),
    )
    request = AzureChatCompletionRequest(messages=[], tools=[tool])
    config = StaticToolsConfig.from_request(request)

    with pytest.raises(
        ValidationError, match="Unsupported static function: unknown_function"
    ):
        config.to_gemini_tools()
