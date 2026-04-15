import json

import httpx
import respx
from google.genai.client import Client as GenAIClient
from google.genai.types import HttpOptions, ResourceScope

from aidial_adapter_vertexai import upstream_config
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    chat_completion,
    function_to_tool,
    user,
)


async def test_streaming_tool_call_finish_reason_is_not_overwritten(
    get_openai_client, monkeypatch
):
    expected_path = (
        "/publishers/google/models/gemini-3-flash-preview:streamGenerateContent"
    )
    upstream_url = f"https://mock-vertex.test{expected_path}"

    chunks = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_temperature",
                                    "args": {
                                        "location": "London",
                                        "unit": "celsius",
                                    },
                                }
                            }
                        ],
                    }
                }
            ]
        },
        {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    ]
    sse_body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)

    http_client = httpx.AsyncClient()
    genai_client = GenAIClient(
        vertexai=True,
        http_options=HttpOptions(
            base_url="https://mock-vertex.test",
            base_url_resource_scope=ResourceScope.COLLECTION,
            httpx_async_client=http_client,
        ),
    )

    async def get_mock_genai_client(project: str, location: str) -> GenAIClient:
        return genai_client

    monkeypatch.setattr(
        upstream_config, "get_genai_client", get_mock_genai_client
    )

    try:
        with respx.mock(assert_all_called=True, assert_all_mocked=True) as rsps:
            route = rsps.post(upstream_url, params={"alt": "sse"}).mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body.encode(),
                )
            )

            client = get_openai_client(D.GEMINI_3_FLASH_PREVIEW.value)
            response = await chat_completion(
                client,
                stream=True,
                messages=[
                    user(
                        "Tell me what's the temperature in London, UK in celsius?"
                    )
                ],
                tools=[function_to_tool(GET_WEATHER_FUNCTION)],
            )
    finally:
        await genai_client.aio.aclose()

    assert route.called
    assert response.finish_reasons == ["tool_calls"]
    assert response.tool_calls is not None
    assert response.tool_calls[0].function.name == "get_temperature"
