import json

import httpx
import respx

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    chat_completion,
    function_to_tool,
    user,
)


async def test_streaming_tool_call_finish_reason_is_not_overwritten(
    get_openai_client,
):
    upstream_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3-flash-preview:streamGenerateContent"
    )

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

    with respx.mock(assert_all_called=True, assert_all_mocked=True) as rsps:
        route = rsps.post(upstream_url, params={"alt": "sse"}).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body.encode(),
            )
        )

        client = get_openai_client(
            D.GEMINI_3_FLASH_PREVIEW.value,
            # Avoiding complicated auth in VertexAI by
            # using the API key which enables Gemini Platform API.
            extra_headers={"x-upstream-key": "test-gemini-api-key"},
        )
        response = await chat_completion(
            client,
            stream=True,
            messages=[
                user("Tell me what's the temperature in London, UK in celsius?")
            ],
            tools=[function_to_tool(GET_WEATHER_FUNCTION)],
        )

    assert route.called
    assert response.finish_reasons == ["tool_calls"]
    assert response.tool_calls is not None
    assert response.tool_calls[0].function.name == "get_temperature"
