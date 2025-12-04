from typing import List

import pydantic
from aidial_sdk.chat_completion import Message as DialMessage
from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart
from pydantic import BaseModel

from aidial_adapter_vertexai.utils.log_config import app_logger as log


class _StateModel(BaseModel):
    model_config = pydantic.ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class Part(_StateModel):
    """Structurally mirrors GenAIPart"""

    thought_signature: bytes | None = None

    def update(self, part: GenAIPart) -> None:
        part.thought_signature = (
            part.thought_signature or self.thought_signature
        )


class Content(_StateModel):
    """Structurally mirrors GenAIContent"""

    parts: List[Part] | None = None

    def update(self, content: GenAIContent) -> None:
        if self.parts and (parts := content.parts):
            for i in range(min(len(self.parts), len(parts))):
                self.parts[i].update(parts[i])


class MessageState(_StateModel):
    gemini_message_content: Content | None = None

    def set_thought_signature(self, thought_signature: bytes) -> None:
        if not self.gemini_message_content:
            self.gemini_message_content = Content(
                parts=[Part(thought_signature=thought_signature)]
            )

    def to_json(self) -> dict:
        return self.model_dump(exclude_none=True, mode="json")


def _parse_message_content_from_state(
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
):
    state = _parse_message_content_from_state(idx, message)

    if state and state.gemini_message_content:
        state.gemini_message_content.update(content)
    else:
        for part in content.parts or []:
            if part.function_call is not None:
                # Last resort if thought_signature wasn't provided via state
                part.thought_signature = b"skip_thought_signature_validator"
                break
