import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    ChatCompletionRequest,
    Function,
    StaticFunction,
    StaticTool,
    Tool,
)
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions
from google.oauth2.credentials import Credentials

from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.adapter import (
    GeminiGenAIChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPromptGenAI
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.utils.adapter_deployment import AdapterDeployment


def test_no_tools():
    request = AzureChatCompletionRequest(messages=[])
    config = StaticToolsConfig.from_request(request)
    assert config.functions == []


class _SpyTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.bodies: list[dict[str, Any]] = []

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [], "totalTokens": 1})


def _http_options(transport: _SpyTransport) -> HttpOptions:
    return HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=transport)
    )


def _gemini_api_client(transport: _SpyTransport) -> GenAIClient:
    return GenAIClient(
        api_key="test-key", http_options=_http_options(transport)
    )


def _vertex_ai_client(transport: _SpyTransport) -> GenAIClient:
    return GenAIClient(
        vertexai=True,
        project="test-project",
        location="us-central1",
        credentials=Credentials(token="test-token"),  # noqa: S106
        http_options=_http_options(transport),
    )


async def _parse(
    client: GenAIClient, function_names: list[str], static_names: list[str]
) -> tuple[GeminiGenAIChatCompletionAdapter, GeminiPromptGenAI]:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],  # type: ignore
        tools=[
            *[
                Tool(type="function", function=Function(name=name))
                for name in function_names
            ],
            *[
                StaticTool(
                    type="static_function",
                    static_function=StaticFunction(name=name),
                )
                for name in static_names
            ],
        ]
        or None,
    )
    adapter = GeminiGenAIChatCompletionAdapter(
        file_storage=None,
        deployment=AdapterDeployment(
            upstream_deployment_id="gemini-2.5-flash",
            reference_deployment_id=ChatCompletionDeployment.GEMINI_2_5_FLASH,
        ),
        client=client,
        headers={},
    )
    prompt = await adapter.parse_prompt(
        ModelParameters.create(request),
        ToolsConfig.from_request(request),
        StaticToolsConfig.from_request(request),
        request,
    )
    assert not isinstance(prompt, UserError)
    return adapter, prompt


def _sent_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    # The order of tools doesn't matter, function declarations are compared by name
    tools = [
        {
            key: (
                [decl["name"] for decl in value]
                if key == "functionDeclarations"
                else value
            )
            for key, value in tool.items()
        }
        for tool in body.get("tools", [])
    ]
    return sorted(tools, key=json.dumps)


_TOOLS_CASES = pytest.mark.parametrize(
    ("function_names", "static_names", "expected_tools"),
    [
        pytest.param([], [], [], id="no_tools"),
        pytest.param(
            ["weather"],
            [],
            [{"functionDeclarations": ["weather"]}],
            id="function",
        ),
        pytest.param(
            [], ["google_search"], [{"googleSearch": {}}], id="static"
        ),
        pytest.param(
            [],
            ["google_search", "code_execution"],
            [{"googleSearch": {}}, {"codeExecution": {}}],
            id="statics",
        ),
        pytest.param(
            ["weather"],
            ["google_search"],
            [{"functionDeclarations": ["weather"]}, {"googleSearch": {}}],
            id="function+google_search",
        ),
        pytest.param(
            ["weather"],
            ["code_execution"],
            [{"codeExecution": {}}, {"functionDeclarations": ["weather"]}],
            id="function+code_execution",
        ),
        pytest.param(
            ["weather", "news"],
            ["google_search", "code_execution"],
            [
                {"functionDeclarations": ["weather", "news"]},
                {"googleSearch": {}},
                {"codeExecution": {}},
            ],
            id="functions+statics",
        ),
    ],
)


@_TOOLS_CASES
@pytest.mark.parametrize(
    "make_client",
    [_gemini_api_client, _vertex_ai_client],
    ids=["gemini_api", "vertex_ai"],
)
async def test_tools_sent_to_generate_content(
    make_client: Callable[[_SpyTransport], GenAIClient],
    function_names: list[str],
    static_names: list[str],
    expected_tools: list[dict[str, Any]],
):
    transport = _SpyTransport()
    adapter, prompt = await _parse(
        make_client(transport), function_names, static_names
    )
    params = ModelParameters()

    async for _ in adapter.send_message_async(params, prompt):
        pass

    [body] = transport.bodies
    assert _sent_tools(body) == sorted(expected_tools, key=json.dumps)


# The Gemini API doesn't support tools in count_tokens
@_TOOLS_CASES
async def test_tools_sent_to_count_tokens(
    function_names: list[str],
    static_names: list[str],
    expected_tools: list[dict[str, Any]],
):
    transport = _SpyTransport()
    adapter, prompt = await _parse(
        _vertex_ai_client(transport), function_names, static_names
    )

    await adapter.count_prompt_tokens(prompt)

    [body] = transport.bodies
    assert _sent_tools(body) == sorted(expected_tools, key=json.dumps)
