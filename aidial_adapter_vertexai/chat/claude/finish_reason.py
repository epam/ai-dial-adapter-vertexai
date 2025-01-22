from typing import Literal, assert_never

from aidial_sdk.chat_completion import FinishReason

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsMode

ClaudeFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use"
]


def to_dial_finish_reason(
    finish_reason: ClaudeFinishReason | None,
    tools_mode: ToolsMode | None,
) -> FinishReason:
    if finish_reason is None:
        return FinishReason.STOP

    match finish_reason:
        case "end_turn":
            return FinishReason.STOP
        case "max_tokens":
            return FinishReason.LENGTH
        case "stop_sequence":
            return FinishReason.STOP
        case "tool_use":
            match tools_mode:
                case ToolsMode.TOOLS:
                    return FinishReason.TOOL_CALLS
                case ToolsMode.FUNCTIONS:
                    return FinishReason.FUNCTION_CALL
                case None:
                    raise ValidationError(
                        "A model has called a tool, but no tools were given to the model in the first place."
                    )
                case _:
                    assert_never(tools_mode)

        case _:
            assert_never(finish_reason)
