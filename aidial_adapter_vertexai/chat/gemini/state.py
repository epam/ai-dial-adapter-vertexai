import pydantic
from aidial_sdk.chat_completion import Message as DialMessage
from google.genai.types import Content as GenAIContent
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


class Content(_StateModel):
    """Structurally mirrors GenAIContent"""

    parts: list[Part] | None = None


class MessageState(_StateModel):
    gemini_message_content: Content | None = None

    def set_thought_signature(self, thought_signature: bytes) -> None:
        if self.gemini_message_content:
            log.warning(
                "Multiple thought_signature's were received within a single Gemini response. "
                "Only the first one will be taken into account."
            )
            return

        self.gemini_message_content = Content(
            parts=[Part(thought_signature=thought_signature)]
        )

    def _get_thought_signature(self) -> bytes | None:
        if self.gemini_message_content is None:
            return None

        parts = self.gemini_message_content.parts
        if parts is None or len(parts) == 0:
            return None

        if len(parts) > 1:
            log.warning(
                "Multiple thought_signature's were received within a single assistant message. "
                "Only the first one will be taken into account."
            )
        return parts[0].thought_signature

    def _disable_thought_signature_validation(self, content: GenAIContent):
        for part in content.parts or []:
            if part.function_call:
                part.thought_signature = b"skip_thought_signature_validator"
                log.warning(
                    "Cannot find thought_signature for a function call block. "
                    "Defaulting to a fake thought_signature."
                )
                return

    def _set_thought_signature(
        self, content: GenAIContent, thought_signature: bytes
    ):
        content_parts = content.parts or []

        # Attach to the first function block if there are any
        for part in content_parts:
            if part.function_call:
                part.thought_signature = thought_signature
                return

        # Otherwise, attach to the last block
        if not content_parts:
            content_parts[-1].thought_signature = thought_signature

    def update_content(self, content: GenAIContent):
        """
        As per documentation:
        https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures/#using-rest-or-manual-handling
        The thought_signature's are integrated to the content blocks in the following way:

        1. If there are no thought_signature's and there are function blocks, then it will result in
            the 400 error from the upstream: content block is missing a `thought_signature`.
            Therefore, we guard against it by setting a fake signature to relax this validation.
        2. If there are any function call blocks, attach the thought signature to the *first* function block.
        3. If there are no function call blocks, attach the thought signature to the *last* block.
        """
        thought_signature = self._get_thought_signature()

        if thought_signature is None:
            self._disable_thought_signature_validation(content)
        else:
            self._set_thought_signature(content, thought_signature)

    def to_json(self) -> dict:
        return self.model_dump(exclude_none=True, mode="json")


def _parse_message_content_from_state(
    idx: int, message: DialMessage
) -> MessageState:
    if (cc := message.custom_content) and (state := cc.state):
        try:
            return MessageState.model_validate(state)
        except pydantic.ValidationError as e:
            log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )

    return MessageState()


def update_with_message_state(
    idx: int, message: DialMessage, content: GenAIContent
):
    state = _parse_message_content_from_state(idx, message)
    state.update_content(content)
