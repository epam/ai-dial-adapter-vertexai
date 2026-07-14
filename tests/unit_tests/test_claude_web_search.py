import pytest
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticFunction,
    StaticTool,
)

from aidial_adapter_vertexai.chat.claude.adapter import _to_configuration
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "web_search_20250305",
    "max_uses": 5,
    "allowed_domains": ["example.com", "example.org"],
    "user_location": {
        "type": "approximate",
        "city": "San Francisco",
        "region": "California",
        "country": "US",
        "timezone": "America/Los_Angeles",
    },
}


def _web_search_static_tools(configuration: dict | None) -> StaticToolsConfig:
    request = AzureChatCompletionRequest(
        messages=[],
        tools=[
            StaticTool(
                type="static_function",
                static_function=StaticFunction(
                    name="web_search",
                    configuration=configuration,
                ),
            )
        ],
    )
    return StaticToolsConfig.from_request(request)


def test_web_search_static_tool_converted_to_configuration():
    static_tools = _web_search_static_tools({"type": "web_search_20250305"})

    configuration = _to_configuration(ModelParameters(), static_tools)

    assert configuration == {
        "web_search": {"type": "web_search_20250305", "name": "web_search"}
    }


def test_web_search_optional_fields_preserved():
    static_tools = _web_search_static_tools(WEB_SEARCH_TOOL_DEFINITION)

    configuration = _to_configuration(ModelParameters(), static_tools)

    assert configuration is not None
    assert configuration["web_search"] == {
        **WEB_SEARCH_TOOL_DEFINITION,
        "name": "web_search",
    }


def test_web_search_merged_with_existing_configuration():
    static_tools = _web_search_static_tools({"type": "web_search_20250305"})
    params = ModelParameters(configuration={"enable_citations": True})

    configuration = _to_configuration(params, static_tools)

    assert configuration == {
        "enable_citations": True,
        "web_search": {"type": "web_search_20250305", "name": "web_search"},
    }


def test_no_static_tools_passes_configuration_through():
    static_tools = StaticToolsConfig.noop()
    params = ModelParameters(configuration={"enable_citations": True})

    configuration = _to_configuration(params, static_tools)

    assert configuration == {"enable_citations": True}


def test_unsupported_static_function_rejected():
    request = AzureChatCompletionRequest(
        messages=[],
        tools=[
            StaticTool(
                type="static_function",
                static_function=StaticFunction(name="google_search"),
            )
        ],
    )
    static_tools = StaticToolsConfig.from_request(request)

    with pytest.raises(ValidationError, match="Unsupported static function"):
        _to_configuration(ModelParameters(), static_tools)


def test_web_search_configured_twice_rejected():
    static_tools = _web_search_static_tools({"type": "web_search_20250305"})
    params = ModelParameters(
        configuration={"web_search": {"type": "web_search_20250305"}}
    )

    with pytest.raises(ValidationError, match="not both"):
        _to_configuration(params, static_tools)
