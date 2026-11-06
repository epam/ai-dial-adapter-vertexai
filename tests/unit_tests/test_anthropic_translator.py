from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

_CORE = "http://dial-core"


async def test_2_plus_3(
    monkeypatch: pytest.MonkeyPatch, test_http_client: httpx.AsyncClient
) -> None:
    import aidial_adapter_vertexai.app as app_module

    core_requests: list[httpx.Request] = []

    def dial_core(request: httpx.Request) -> httpx.Response:
        core_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "5"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setenv("DIAL_URL", _CORE)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(dial_core)) as core,
        anthropic.AsyncAnthropic(
            api_key="dummy-key",
            base_url="http://test-app.com/openaiChatCompletions/anthropic",
            http_client=test_http_client,
            max_retries=0,
        ) as client,
    ):
        monkeypatch.setattr(
            app_module, "get_httpx_client", AsyncMock(return_value=core)
        )
        response = await client.messages.create(
            model="gpt-5.5",
            max_tokens=100,
            messages=[{"role": "user", "content": "2+3=?"}],
        )

    assert [str(r.url) for r in core_requests] == [
        f"{_CORE}/openai/v1/chat/completions"
    ]
    assert response.content[0].type == "text"
    assert response.content[0].text == "5"
