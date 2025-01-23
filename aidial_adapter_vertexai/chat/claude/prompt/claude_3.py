from typing import List, Set, Tuple

from aidial_sdk.chat_completion import Message
from anthropic.types import MessageParam, TextBlockParam

from aidial_adapter_vertexai.chat.claude.conversation_factory import (
    SUPPORTED_IMAGE_TYPES,
    ClaudeConversationFactory,
    ClaudePart,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.chat.conversation.inputs import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessor,
    AttachmentProcessorsBase,
    max_count_validator,
    seq_validators,
)
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    get_usage_message,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.list import group_by
from aidial_adapter_vertexai.utils.list_projection import ListProjection


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
        processors=[create_image_processor(20)] if supports_vision else [],
        file_storage=file_storage,
    )

    conversation = await messages_to_conversation(
        conversation_factory, processors, tools, messages
    )

    conversation = conversation.on_messages(_merge_messages_with_same_role)

    if error_message := processors.get_error_message():
        usage_message = get_usage_message(processors.get_file_exts())
        return UserError(error_message, usage_message)

    return ClaudePrompt(conversation=conversation, tools=tools)


def _merge_messages_with_same_role(
    messages: ListProjection[MessageParam],
) -> ListProjection[MessageParam]:
    def _key(message: Tuple[MessageParam, Set[int]]) -> str:
        return message[0]["role"]

    def _merge(
        a: Tuple[MessageParam, Set[int]],
        b: Tuple[MessageParam, Set[int]],
    ) -> Tuple[MessageParam, Set[int]]:
        (msg1, set1), (msg2, set2) = a, b

        content1 = msg1["content"]
        content2 = msg2["content"]

        if isinstance(content1, str):
            content1 = [TextBlockParam(type="text", text=content1)]

        if isinstance(content2, str):
            content2 = [TextBlockParam(type="text", text=content2)]

        return (
            MessageParam(
                role=msg1["role"], content=list(content1) + list(content2)
            ),
            set1 | set2,
        )

    return ListProjection(group_by(messages.list, _key, lambda x: x, _merge))


def create_image_processor(max_count: int) -> AttachmentProcessor:
    # NOTE: not checked condition: The maximum allowed image file size is 5 MB
    return AttachmentProcessor(
        file_types=SUPPORTED_IMAGE_TYPES,
        init_validator=seq_validators(None, max_count_validator(max_count)),
    )
