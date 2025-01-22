from typing import List, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.claude.conversation_factory import (
    ClaudeConversationFactory,
    ClaudePart,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.chat.conversation.inputs import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessorsBase,
)
from aidial_adapter_vertexai.chat.gemini.processors import get_image_processor
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    get_usage_message,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class AttachmentProcessorsClaude(AttachmentProcessorsBase[ClaudePart]):
    pass


class Claude_3_Prompt(ClaudePrompt):
    @classmethod
    async def parse(
        cls,
        file_storage: FileStorage | None,
        tools: ToolsConfig,
        messages: List[Message],
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        conversation_factory = ClaudeConversationFactory()

        processors = AttachmentProcessorsClaude(
            conversation_factory=conversation_factory,
            processors=[
                # NOTE: not checked condition: The maximum allowed image file size is 5 MB
                get_image_processor(20),
            ],
            file_storage=file_storage,
        )

        conversation = await messages_to_conversation(
            conversation_factory, processors, tools, messages
        )

        if error_message := processors.get_error_message():
            usage_message = get_usage_message(processors.get_file_exts())
            return UserError(error_message, usage_message)

        return cls(conversation=conversation, tools=tools)
