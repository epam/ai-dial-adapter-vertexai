from typing import List, Optional, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.conversation.converters import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.conversation_factory import (
    ConversationFactoryGenAI,
)
from aidial_adapter_vertexai.chat.gemini.processors import (
    GeminiAttachmentProcessorsGenAI,
    get_audio_processor,
    get_image_processor,
    get_pdf_processor,
    get_plain_text_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPromptGenAI
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    get_usage_message,
)
from aidial_adapter_vertexai.chat.gemini.prompt.message import (
    GenAIMessageMerger,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_2_Prompt(GeminiPromptGenAI):
    @classmethod
    async def parse(
        cls,
        file_storage: Optional[FileStorage],
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        conversation_factory = ConversationFactoryGenAI()
        # TODO: update limits, when they are published
        processors = GeminiAttachmentProcessorsGenAI(
            conversation_factory=conversation_factory,
            processors=[
                get_plain_text_processor(),
                get_image_processor(3000),
                get_pdf_processor(
                    page_limit_per_request=3000,
                    page_limit_per_document=1000,
                ),
                get_video_processor(10),
                get_audio_processor(),
            ],
            file_storage=file_storage,
        )

        conversation = await messages_to_conversation(
            conversation_factory, processors, tools, messages
        )
        conversation = conversation.merge_messages_with_same_role(
            GenAIMessageMerger
        )

        if error_message := processors.get_error_message():
            usage_message = get_usage_message(processors.get_file_exts())
            return UserError(error_message, usage_message)

        return cls(
            conversation=conversation, tools=tools, static_tools=static_tools
        )
