from typing import List, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.attachment_processor import (
    AttachmentProcessors,
)
from aidial_adapter_vertexai.chat.conversation.converters import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.conversation_factory import (
    ConversationFactory,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig


class Gemini_1_0_Pro_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        conversation_factory = ConversationFactory()
        processors = AttachmentProcessors(
            conversation_factory=conversation_factory,
            processors=[],
            file_storage=None,
        )

        conversation = await messages_to_conversation(
            conversation_factory, processors, tools, messages
        )

        if error_message := processors.get_error_message():
            return UserError(error_message, error_message)

        return cls(
            conversation=conversation, tools=tools, static_tools=static_tools
        )
