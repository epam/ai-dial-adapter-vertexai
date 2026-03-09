from typing import assert_never

from aidial_sdk.chat_completion.request import ReasoningEffort
from google.genai.types import ThinkingLevel


def to_genai_thinking_level(
    reasoning_effort: ReasoningEffort | None,
) -> ThinkingLevel | None:
    """
    Follows the official mapping to OpenAI API:
    https://ai.google.dev/gemini-api/docs/gemini-3?thinking=low#openai_compatibility
    """

    match reasoning_effort:
        case None | ReasoningEffort.NONE:
            return None
        case ReasoningEffort.MINIMAL:
            return ThinkingLevel.MINIMAL
        case ReasoningEffort.LOW:
            return ThinkingLevel.LOW
        case ReasoningEffort.MEDIUM:
            return ThinkingLevel.MEDIUM
        case ReasoningEffort.HIGH:
            return ThinkingLevel.HIGH
        case _:
            assert_never(reasoning_effort)
