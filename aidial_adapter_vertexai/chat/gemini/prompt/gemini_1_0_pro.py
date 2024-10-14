from typing import List, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.inputs import (
    messages_to_gemini_content,
)
from aidial_adapter_vertexai.chat.gemini.processor import AttachmentProcessors
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.tools import ToolsConfig


class Gemini_1_0_Pro_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls, tools: ToolsConfig, messages: List[Message]
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        processors = AttachmentProcessors(processors=[], file_storage=None)

        history = await messages_to_gemini_content(processors, tools, messages)

        if error_message := processors.get_error_message():
            return UserError(error_message, error_message)

        return cls(
            history=history[:-1],
            prompt=history[-1].parts,
            tools=tools,
        )
