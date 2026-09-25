from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    ReasoningEffort,
    StaticFunction,
    StaticTool,
)

from aidial_adapter_vertexai.chat.claude.adapter import _to_adapter_request
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters

WEB_SEARCH_CONFIGURATION = {
    "type": "web_search_20250305",
    "max_uses": 5,
    "allowed_domains": ["example.com", "example.org"],
}


def test_web_search_static_tool_passed_through_to_sdk():
    web_search = StaticTool(
        type="static_function",
        static_function=StaticFunction(
            name="web_search",
            configuration=WEB_SEARCH_CONFIGURATION,
        ),
    )
    request = ChatCompletionRequest(messages=[], tools=[web_search])
    tools = ToolsConfig.from_request(request)
    static_tools = StaticToolsConfig.from_request(request)

    params = _to_adapter_request(
        request, ModelParameters(), tools, static_tools
    )

    assert params.tool_config is not None
    assert params.tool_config.tools == []
    assert params.tool_config.static_tools == [web_search]
    assert params.configuration is None


def test_no_static_tools_keeps_empty_static_tools_list():
    request = ChatCompletionRequest(messages=[])
    tools = ToolsConfig.noop()
    static_tools = StaticToolsConfig.noop()

    params = _to_adapter_request(
        request,
        ModelParameters(configuration={"enable_citations": True}),
        tools,
        static_tools,
    )

    assert params.tool_config is not None
    assert params.tool_config.static_tools == []
    assert params.configuration == {"enable_citations": True}


def test_request_fields_forwarded_to_sdk():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
            "custom_fields": {"cache_breakpoint": {"type": "ephemeral"}},
        }
    )

    params = _to_adapter_request(
        request, ModelParameters(), ToolsConfig.noop(), StaticToolsConfig.noop()
    )

    assert request.custom_fields is not None
    assert params.reasoning_effort is ReasoningEffort.HIGH
    assert params.response_format == request.response_format
    assert params.cache_breakpoint == request.custom_fields.cache_breakpoint
