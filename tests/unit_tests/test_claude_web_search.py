from aidial_adapter_anthropic.dial.request import AdapterRequest
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    ReasoningEffort,
)

_WEB_SEARCH_CONFIGURATION = {
    "type": "web_search_20250305",
    "max_uses": 5,
    "allowed_domains": ["example.com", "example.org"],
}


def test_web_search_static_tool_passed_through_to_sdk():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [],
            "tools": [
                {
                    "type": "static_function",
                    "static_function": {
                        "name": "web_search",
                        "configuration": _WEB_SEARCH_CONFIGURATION,
                    },
                }
            ],
        }
    )
    params = AdapterRequest.create(request)

    assert params.tool_config is not None
    assert params.tool_config.tools == []
    assert len(params.tool_config.static_tools) == 1
    tool = params.tool_config.static_tools[0]
    assert tool.static_function.name == "web_search"
    assert tool.static_function.configuration == _WEB_SEARCH_CONFIGURATION
    assert params.configuration is None


def test_no_static_tools_keeps_empty_static_tools_list():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [],
            "custom_fields": {"configuration": {"enable_citations": True}},
        }
    )

    params = AdapterRequest.create(request)

    assert params.tool_config is None
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

    params = AdapterRequest.create(request)

    assert request.custom_fields is not None
    assert params.reasoning_effort is ReasoningEffort.HIGH
    assert params.response_format == request.response_format
    assert params.cache_breakpoint == request.custom_fields.cache_breakpoint
