from typing import Iterable, List, TypedDict, TypeVar, Union

from anthropic import Omit, omit
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolParam as ToolParam

from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.utils.env import get_env_int

_DEFAULT_MAX_TOKENS = get_env_int("CLAUDE_DEFAULT_MAX_TOKENS", 1536)


_T = TypeVar("_T")


def none_to_omit(value: _T | None) -> _T | Omit:
    return value if value is not None else omit


class ChatParameters(TypedDict):
    max_tokens: int
    stop_sequences: List[str] | Omit
    temperature: float | Omit
    top_p: float | Omit

    tools: List[ToolParam] | Omit
    tool_choice: ToolChoice | Omit
    system: Union[str, Iterable[TextBlockParam]] | Omit
    betas: List[AnthropicBetaParam] | Omit


def create_chat_params(
    params: ModelParameters,
    prompt: ClaudePrompt,
    betas: List[AnthropicBetaParam] | None,
) -> ChatParameters:
    system = none_to_omit(prompt.system)
    tools = none_to_omit(prompt.tools.to_claude_tools())
    tool_choice = none_to_omit(prompt.tools.to_claude_tool_choice())

    temperature = none_to_omit(params.temperature)
    stop_sequences = none_to_omit(params.stop)
    max_tokens = params.max_tokens or _DEFAULT_MAX_TOKENS
    top_p = none_to_omit(params.top_p)

    return ChatParameters(
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        top_p=top_p,
        betas=betas or omit,
    )
