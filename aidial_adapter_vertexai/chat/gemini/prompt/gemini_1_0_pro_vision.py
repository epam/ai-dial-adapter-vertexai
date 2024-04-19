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

        # NOTE: only a single message is supported by Gemini 1.0 Pro Vision,
        # when non-text parts are present in the request.
        #
        # Otherwise, the following error is returned by VertexAI:
        #   400 Unable to submit request because it has more than one contents field but model gemini-pro-vision only supports one. Remove all but one contents and try again.
        #
        # Conversely, multi-turn chat with text-only content is supported.
        # However, we disable it, otherwise, it's hard to make it clear to the user,
        # when the model works in a single-turn mode and when in a multi-turn mode.
        # Thus, we make the model single-turn altogether.
        messages = messages[-1:]

        exclusive = exclusive_validator()
        processors = [
            get_image_processor(16, exclusive("image")),
            get_pdf_processor(16, exclusive("pdf")),
            get_video_processor(1, exclusive("video")),
        ]

        result = await process_messages(processors, file_storage, messages)

        usage_message = get_usage_message(get_file_exts(processors))

        if isinstance(result, str):
            return UserError(result, usage_message)

        if all(len(res.resources) == 0 for res in result):
            return UserError("No documents were found", usage_message)

        history = [res.to_content() for res in result]
        return cls(history=history[:-1], prompt=history[-1].parts)


def get_usage_message(exts: List[str]) -> str:
    return f"""
### Usage

The application answers queries about attached documents.
Attach documents and ask questions about them in the same message.

Only the last message will be taken into account.

The images, PDFs and videos must not be mixed in the same message.

Supported document extensions: {', '.join(exts)}.

Examples of queries:
- "Describe the picture" for one image,
- "What is depicted in these images?", "Compare the images" for multiple images,
- "Summarize the document" for a PDF,
- "What is happening in the video?" for a video.
""".strip()
