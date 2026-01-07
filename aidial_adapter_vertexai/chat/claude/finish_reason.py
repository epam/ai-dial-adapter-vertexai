from typing import assert_never

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from anthropic.types.beta import BetaStopReason as ClaudeFinishReason

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsMode


def to_dial_finish_reason(
    finish_reason: ClaudeFinishReason | None,
    tools_mode: ToolsMode | None,
) -> DialFinishReason:
    if finish_reason is None:
        return DialFinishReason.STOP

    match finish_reason:
        case "end_turn" | "pause_turn" | "refusal" | "stop_sequence":
            return DialFinishReason.STOP
        case "max_tokens" | "model_context_window_exceeded":
            return DialFinishReason.LENGTH
        case "tool_use":
            match tools_mode:
                case ToolsMode.TOOLS:
                    return DialFinishReason.TOOL_CALLS
                case ToolsMode.FUNCTIONS:
                    return DialFinishReason.FUNCTION_CALL
                case None:
                    raise ValidationError(
                        "A model has called a tool, but no tools were given to the model in the first place."
                    )
                case _:
                    assert_never(tools_mode)
        case _:
            assert_never(finish_reason)
