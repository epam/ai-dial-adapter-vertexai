from typing import List, Optional, Self, Union

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import process_messages
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_audio_processor,
    get_file_exts,
    get_image_processor,
    get_pdf_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_0_pro import (
    accommodate_first_system_message,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_1_5_Pro_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls,
        file_storage: Optional[FileStorage],
        tools: ToolsConfig,
        messages: List[Message],
    ) -> Union[Self, UserError]:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        messages = accommodate_first_system_message(messages)

        processors = [
            get_image_processor(3000),
            get_pdf_processor(3000),
            get_video_processor(10),
            get_audio_processor(),
        ]

        result = await process_messages(processors, file_storage, messages)

        if isinstance(result, str):
            usage_message = get_usage_message(get_file_exts(processors))
            return UserError(result, usage_message)

        history = [res.to_content() for res in result]
        return cls(
            history=history[:-1],
            prompt=history[-1].parts,
            tools=tools,
        )


def get_usage_message(exts: List[str]) -> str:
    return f"""
### Usage

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
