from typing import List, Optional, Self, Union

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    exclusive_validator,
    process_messages,
)
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_file_exts,
    get_image_processor,
    get_pdf_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_1_0_Pro_Vision_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls, file_storage: Optional[FileStorage], messages: List[Message]
    ) -> Union[Self, UserError]:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        exclusive = exclusive_validator()
        processors = [
            get_image_processor(16, exclusive("image")),
            get_pdf_processor(16, exclusive("pdf")),
            get_video_processor(1, exclusive("video")),
        ]

        download_result = await process_messages(
            processors, file_storage, messages
        )

        usage_message = get_usage_message(get_file_exts(processors))

        if isinstance(download_result, str):
            return UserError(download_result, usage_message)

        history = [res.to_content() for res in download_result]
        return cls(history=history[:-1], prompt=history[-1].parts)


def get_usage_message(exts: List[str]) -> str:
    return f"""
### Usage

The application answers queries about attached documents.
Attach documents and ask questions about them in the same message.

Supported document types: {', '.join(exts)}.

The images, PDFs and videos must not be mixed in the same message.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
