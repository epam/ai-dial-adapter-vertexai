from typing import Any, cast

import pytest
from aidial_sdk.chat_completion import FinishReason, Request, Response

from aidial_adapter_vertexai.chat.consumer import ChoiceConsumer


@pytest.mark.parametrize(
    "finish_reason", [FinishReason.LENGTH, FinishReason.STOP]
)
async def test_streaming_tool_call_finish_reason_is_not_overwritten(
    finish_reason: FinishReason,
):
    response = Response(Request.model_construct(stream=True, n=1))

    with ChoiceConsumer(response=response) as consumer:
        await consumer.create_tool_call(
            id="call_id",
            name="get_temperature",
            arguments='{"location": "London", "unit": "celsius"}',
        )
        await consumer.set_finish_reason(finish_reason)
        assert consumer.get_finish_reason() == FinishReason.TOOL_CALLS

    chunks = []
    while not response._queue.empty():
        chunk = cast(Any, response._queue.get_nowait())
        chunks.append(chunk.to_dict())
        response._queue.task_done()

    assert (
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        == "get_temperature"
    )
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"
