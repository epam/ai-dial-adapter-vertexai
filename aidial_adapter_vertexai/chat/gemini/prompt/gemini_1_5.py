from typing import List, Optional, Self

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.inputs import (
    messages_to_gemini_conversation,
)
from aidial_adapter_vertexai.chat.gemini.processor import AttachmentProcessors
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_audio_processor,
    get_image_processor,
    get_pdf_processor,
    get_plain_text_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_1_5_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls,
        file_storage: Optional[FileStorage],
        tools: ToolsConfig,
        messages: List[Message],
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        processors = AttachmentProcessors(
            processors=[
                get_plain_text_processor(),
                get_image_processor(3000),
                get_pdf_processor(300),
                get_video_processor(10),
                get_audio_processor(),
            ],
            file_storage=file_storage,
        )

        conversation = await messages_to_gemini_conversation(
            processors, tools, messages
        )

        if error_message := processors.get_error_message():
            usage_message = get_usage_message(processors.get_file_exts())
            return UserError(error_message, usage_message)

        return cls(
            system_instruction=conversation.system_instruction,
            contents=conversation.contents,
            tools=tools,
        )


def get_usage_message(exts: List[str]) -> str:
    return f"""
The application answers queries about attached documents.
Attach documents and ask questions about them in the same message.

Supported document extensions: {', '.join(exts)}.

Examples of queries:
- "Describe the picture" for one image,
- "What is depicted in these images?", "Compare the images" for multiple images,
- "Summarize the document" for a PDF,
- "Transcribe the audio" for an audio file,
- "What is happening in the video?" for a video.
""".strip()
