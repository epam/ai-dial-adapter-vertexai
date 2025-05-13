from typing import Iterable, List, TypedDict, TypeVar, Union

from anthropic import NOT_GIVEN, NotGiven
from anthropic.types import TextBlockParam, ToolParam
from anthropic.types.message_create_params import ToolChoice

from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.dial_api.request import ModelParameters

_DEFAULT_MAX_TOKENS_CLAUDE = 1536


_T = TypeVar("_T")


def none_to_not_given(value: _T | None) -> _T | NotGiven:
    return value if value is not None else NOT_GIVEN


class ChatParameters(TypedDict):
    max_tokens: int
    stop_sequences: List[str] | NotGiven
    temperature: float | NotGiven
    top_p: float | NotGiven

    tools: List[ToolParam] | NotGiven
    tool_choice: ToolChoice | NotGiven
    system: Union[str, Iterable[TextBlockParam]] | NotGiven


def create_chat_params(
    params: ModelParameters, prompt: ClaudePrompt
) -> ChatParameters:
    system = none_to_not_given(prompt.system)
    tools = none_to_not_given(prompt.tools.to_claude_tools())
    tool_choice = none_to_not_given(prompt.tools.to_claude_tool_config())

    temperature = none_to_not_given(params.temperature)
    stop_sequences = none_to_not_given(params.stop)
    max_tokens = params.max_tokens or _DEFAULT_MAX_TOKENS_CLAUDE
    top_p = none_to_not_given(params.top_p)

    return {
        "system": system,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
