import pytest
from aidial_sdk.chat_completion import FinishReason
from google.genai.types import FinishReason as GenAIFinishReason

from aidial_adapter_vertexai.chat.gemini.error import FinishReasonOtherError
from aidial_adapter_vertexai.chat.gemini.finish_reason import (
    genai_to_openai_finish_reason,
    invalid_tool_call_message,
)


@pytest.mark.parametrize(
    ("genai_reason", "expected"),
    [
        (None, None),
        (GenAIFinishReason.FINISH_REASON_UNSPECIFIED, None),
        (GenAIFinishReason.MAX_TOKENS, FinishReason.LENGTH),
        (GenAIFinishReason.STOP, FinishReason.STOP),
        (GenAIFinishReason.SAFETY, FinishReason.CONTENT_FILTER),
        (GenAIFinishReason.RECITATION, FinishReason.CONTENT_FILTER),
        # malformed/unexpected tool call -> terminal finish reason, not a 5xx
        (GenAIFinishReason.MALFORMED_FUNCTION_CALL, FinishReason.STOP),
        (GenAIFinishReason.UNEXPECTED_TOOL_CALL, FinishReason.STOP),
    ],
)
def test_finish_reason_maps_to_terminal_reason(
    genai_reason: GenAIFinishReason | None, expected: FinishReason | None
):
    assert (
        genai_to_openai_finish_reason(genai_reason, None, retriable=True)
        == expected
    )


@pytest.mark.parametrize(
    "retriable",
    [True, False],
)
def test_malformed_function_call_never_raises(retriable: bool):
    # Regression: this case used to raise (surfaced as a 500).
    assert (
        genai_to_openai_finish_reason(
            GenAIFinishReason.MALFORMED_FUNCTION_CALL, None, retriable=retriable
        )
        == FinishReason.STOP
    )


def test_other_finish_reason_still_raises():
    with pytest.raises(FinishReasonOtherError):
        genai_to_openai_finish_reason(
            GenAIFinishReason.OTHER, None, retriable=True
        )


@pytest.mark.parametrize(
    "genai_reason",
    [
        GenAIFinishReason.MALFORMED_FUNCTION_CALL,
        GenAIFinishReason.UNEXPECTED_TOOL_CALL,
    ],
)
def test_invalid_tool_call_message_carries_finish_message(genai_reason):
    # The dropped call is replaced by client-facing text that keeps the detail.
    msg = invalid_tool_call_message(genai_reason, "bad function args")
    assert msg is not None
    assert "bad function args" in msg


def test_invalid_tool_call_message_none_for_other_reasons():
    assert invalid_tool_call_message(GenAIFinishReason.STOP, None) is None
    assert invalid_tool_call_message(None, "x") is None
