from typing import assert_never

from aidial_sdk.chat_completion import FinishReason
from google.genai.types import FinishReason as GenAIFinishReason

from aidial_adapter_vertexai.chat.gemini.error import FinishReasonOtherError

_EARLY_TERMINATION_ERROR = "The model terminated generation unexpectedly"
_DROPPED_TOOL_CALL_MESSAGE = (
    "The model generated an invalid tool call, which was discarded."
)


def genai_to_openai_finish_reason(
    finish_reason: GenAIFinishReason | None,
    finish_message: str | None,
    retriable: bool,
) -> FinishReason | None:
    if not finish_reason:
        return None

    def _add_finish_message(msg: str):
        if finish_message:
            return f"{msg}: {finish_message}"
        else:
            return msg

    match finish_reason:
        case GenAIFinishReason.FINISH_REASON_UNSPECIFIED:
            return None
        case GenAIFinishReason.MAX_TOKENS:
            return FinishReason.LENGTH
        case GenAIFinishReason.STOP:
            return FinishReason.STOP
        case (
            GenAIFinishReason.SAFETY
            | GenAIFinishReason.RECITATION
            | GenAIFinishReason.BLOCKLIST
            | GenAIFinishReason.PROHIBITED_CONTENT
            | GenAIFinishReason.SPII
            | GenAIFinishReason.IMAGE_SAFETY
            | GenAIFinishReason.LANGUAGE
        ):
            return FinishReason.CONTENT_FILTER
        case GenAIFinishReason.OTHER:
            raise FinishReasonOtherError(
                msg=_add_finish_message(_EARLY_TERMINATION_ERROR),
                retriable=retriable,
            )
        case (
            GenAIFinishReason.MALFORMED_FUNCTION_CALL
            | GenAIFinishReason.UNEXPECTED_TOOL_CALL
        ):
            # Model-output condition, not a provider fault: map to a terminal
            # finish reason instead of a 5xx. output.py drops the invalid call;
            # the caller surfaces invalid_tool_call_message in its place.
            return FinishReason.STOP
        case _:
            raise FinishReasonOtherError(
                msg=_add_finish_message(
                    f"Unexpected finish reason: {finish_reason.value}"
                ),
                retriable=retriable,
            )
            assert_never(finish_reason)


def invalid_tool_call_message(
    finish_reason: GenAIFinishReason | None, finish_message: str | None
) -> str | None:
    # Returned to the client in place of a dropped tool call (see
    # genai_to_openai_finish_reason) so the turn carries an explanation rather
    # than empty content and the model can retry.
    if finish_reason not in (
        GenAIFinishReason.MALFORMED_FUNCTION_CALL,
        GenAIFinishReason.UNEXPECTED_TOOL_CALL,
    ):
        return None
    if finish_message:
        return f"{_DROPPED_TOOL_CALL_MESSAGE} {finish_message}"
    return _DROPPED_TOOL_CALL_MESSAGE
