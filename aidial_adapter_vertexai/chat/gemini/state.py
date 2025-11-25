from typing import List

import pydantic
from aidial_sdk.chat_completion import Message as DialMessage
from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from pydantic import BaseModel

from aidial_adapter_vertexai.utils.log_config import app_logger as log


class GenAISubPart(BaseModel):
    thought_signature: bytes | None = None

    def merge(self, part: GenAIPart) -> GenAIPart:
        if self.thought_signature is None:
            return part
        part = part.model_copy()
        part.thought_signature = self.thought_signature
        return part


class GenAISubContent(BaseModel):
    parts: List[GenAISubPart] | None = None

    def merge(self, parts: List[GenAIPart] | None) -> List[GenAIPart] | None:
        if self.parts is None or parts is None:
            return parts

        parts = parts[:]
        for i in range(min(len(self.parts), len(parts))):
            parts[i] = self.parts[i].merge(parts[i])
        return parts


class MessageState(BaseModel):
    gemini_message_content: GenAISubContent


def _get_message_content_from_state(
    idx: int, message: DialMessage
) -> MessageState | None:
    if (cc := message.custom_content) and (state := cc.state):
        try:
            return MessageState.model_validate(state)
        except pydantic.ValidationError as e:
            log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )

    return None


def update_with_message_state(
    idx: int, message: DialMessage, content: GenAIContent
) -> GenAIContent:
    state = _get_message_content_from_state(idx, message)

    if state is not None:
        content.parts = state.gemini_message_content.merge(content.parts)
    else:
        for part in content.parts or []:
            if part.function_call is not None:
                # Last resort if thought_signature wasn't provide via state
                part.thought_signature = b"skip_thought_signature_validator"

    return content
