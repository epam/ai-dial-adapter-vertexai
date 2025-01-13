from typing import List, Optional, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.input import (
    messages_to_gemini_genai_conversation,
)
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessorsGenAI,
)
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_audio_processor,
    get_image_processor,
    get_pdf_processor,
    get_plain_text_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiGenAIPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    get_usage_message,
)
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_2_Prompt(GeminiGenAIPrompt):
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

        processors = AttachmentProcessorsGenAI(
            processors=[
                get_plain_text_processor(),
                get_image_processor(3000),
                get_pdf_processor(300),
                get_video_processor(10),
                get_audio_processor(),
            ],
            file_storage=file_storage,
        )

        conversation = await messages_to_gemini_genai_conversation(
            processors, tools, messages
        )

        if error_message := processors.get_error_message():
            usage_message = get_usage_message(processors.get_file_exts())
            return UserError(error_message, usage_message)

        return cls(
            system_instruction=conversation.system_instruction,
            contents=conversation.contents,
            tools=tools,
            static_tools=static_tools,
        )
