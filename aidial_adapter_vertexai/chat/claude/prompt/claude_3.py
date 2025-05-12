from typing import List

from aidial_sdk.chat_completion import Message
from anthropic.types import MessageParam, TextBlockParam

from aidial_adapter_vertexai.chat.attachment_processor import (
    AttachmentProcessor,
    AttachmentProcessorsBase,
    max_count_validator,
)
from aidial_adapter_vertexai.chat.claude.conversation_factory import (
    SUPPORTED_IMAGE_TYPES,
    ClaudeConversationFactory,
    ClaudePart,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.chat.conversation.converters import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.list import MessageMergeStrategy


class AttachmentProcessorsClaude(AttachmentProcessorsBase[ClaudePart]):
    pass


async def parse_claude_3_prompt(
    file_storage: FileStorage | None,
    tools: ToolsConfig,
    messages: List[Message],
    *,
    supports_vision: bool,
) -> ClaudePrompt | UserError:

    if len(messages) == 0:
        raise ValidationError("The chat history must have at least one message")

    conversation_factory = ClaudeConversationFactory()

    processors = AttachmentProcessorsClaude(
        conversation_factory=conversation_factory,
        processors=[_create_image_processor(20)] if supports_vision else [],
        file_storage=file_storage,
    )

    conversation = await messages_to_conversation(
        conversation_factory, processors, tools, messages
    )
    conversation = conversation.merge_messages_with_same_role(MessageMerger)

    if error_message := processors.get_error_message():
        usage_message = get_usage_message(processors.get_file_exts())
        return UserError(error_message, usage_message)

    return ClaudePrompt(conversation=conversation, tools=tools)


class MessageMerger(MessageMergeStrategy[MessageParam]):
    @staticmethod
    def role(message: MessageParam) -> str:
        return message["role"]

    @staticmethod
    def merge(a: MessageParam, b: MessageParam) -> MessageParam:
        if a["role"] != b["role"]:
            raise ValueError("Cannot merge messages with different roles")

        content1 = a["content"]
        content2 = b["content"]

        if isinstance(content1, str):
            content1 = [TextBlockParam(type="text", text=content1)]

        if isinstance(content2, str):
            content2 = [TextBlockParam(type="text", text=content2)]

        return MessageParam(
            role=a["role"], content=list(content1) + list(content2)
        )


def _create_image_processor(max_count: int) -> AttachmentProcessor:
    # NOTE: not checked condition: The maximum allowed image file size is 5 MB
    return AttachmentProcessor(
        file_types=SUPPORTED_IMAGE_TYPES,
        init_validator=max_count_validator("image", max_count),
    )


def get_usage_message(exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Supported document extensions: {', '.join(exts)}.

Examples of queries:
- "Describe the picture" for one image,
- "What is depicted in these images?", "Compare the images" for multiple images.
""".strip()
