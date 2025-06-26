from typing import List, Literal

from aidial_sdk.chat_completion import Message
from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam

from aidial_adapter_vertexai.chat.attachment_processor import (
    AttachmentProcessor,
    AttachmentProcessorsBase,
)
from aidial_adapter_vertexai.chat.claude.conversation_factory import (
    SUPPORTED_IMAGE_TYPES,
    ClaudeConversationFactory,
    ClaudePart,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import (
    ClaudeMessage,
    ClaudePrompt,
)
from aidial_adapter_vertexai.chat.conversation.converters import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_pdf_processor,
    get_plain_text_processor,
)
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
    enable_citations: bool,
) -> ClaudePrompt | UserError:

    if len(messages) == 0:
        raise ValidationError("The chat history must have at least one message")

    # We do not check the number/sizes of PDFs, images and other attachments.
    # They are constantly changing and different for different models.
    # It's user's responsibility to keep track of the model limits.
    #
    # Image limits: https://docs.anthropic.com/en/docs/build-with-claude/vision#basics-and-limits
    # PDF limits: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support#check-pdf-requirements

    procs: List[AttachmentProcessor] = [
        get_pdf_processor(),
        get_plain_text_processor(),
    ]
    if supports_vision:
        procs.append(get_image_processor())

    conversation_factory = ClaudeConversationFactory(
        enable_citations=enable_citations
    )
    processors = AttachmentProcessorsClaude(
        conversation_factory=conversation_factory,
        processors=procs,
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


class MessageMerger(MessageMergeStrategy[ClaudeMessage]):
    @staticmethod
    def role(message: ClaudeMessage) -> Literal["user", "assistant"]:
        return message.claude_message["role"]

    @staticmethod
    def merge(a: ClaudeMessage, b: ClaudeMessage) -> ClaudeMessage:
        claude_message = MessageMerger.merge_claude_messages(
            a.claude_message, b.claude_message
        )
        dial_messages = a.dial_resources + b.dial_resources
        return ClaudeMessage(
            claude_message=claude_message,
            dial_resources=dial_messages,
        )

    @staticmethod
    def merge_claude_messages(a: MessageParam, b: MessageParam) -> MessageParam:
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


def get_image_processor() -> AttachmentProcessor:
    return AttachmentProcessor(file_types=SUPPORTED_IMAGE_TYPES)


def get_usage_message(exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Supported document extensions: {', '.join(exts)}.

Examples of queries:
- "Describe the picture" for one image,
- "What is depicted in these images?", "Compare the images" for multiple images.
""".strip()
