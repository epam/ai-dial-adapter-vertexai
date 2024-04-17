from typing import Dict, List, Optional, Self, Union

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.process_inputs import download_inputs
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.dial_api.storage import FileStorage

# Gemini capabilities: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts
# Prompt design: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing

FileTypes = Dict[str, Union[str, List[str]]]

# Tokens per image: 258. count_tokens API call takes this into account.
IMAGE_TYPES: FileTypes = {
    "image/jpeg": ["jpg", "jpeg"],
    "image/png": "png",
}
IMAGE_MAX_NUMBER = 16

# PDFs are treated as images, so a single page of a PDF is treated as one image.
PDF_TYPES: FileTypes = {"application/pdf": "pdf"}
PDF_MAX_TOTAL_PAGES = 16  # same as IMAGE_MAX_NUMBER
PDF_MAX_FILE_SIZE_MB = 50

# Audio in the video is ignored.
# Videos are sampled at 1fps. Each video frame accounts for 258 tokens.
# The video is automatically truncated to the first two minutes.
VIDEO_TYPES: FileTypes = {
    "video/mp4": "mp4",
    "video/mov": "mov",
    "video/mpeg": "mpeg",
    "video/mpg": "mpg",
    "video/avi": "avi",
    "video/wmv": "wmv",
    "video/mpegps": "mpegps",
    "video/flv": "flv",
}
VIDEO_MAX_NUMBER = 1


def get_mime_types(types: FileTypes) -> List[str]:
    return list(types.keys())


def get_file_exts(types: FileTypes) -> List[str]:
    def flatten(value: Union[str, List[str]]):
        return value if isinstance(value, list) else [value]

    return [ext for exts in types.values() for ext in flatten(exts)]


class GeminiProOneVisionPrompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls, file_storage: Optional[FileStorage], messages: List[Message]
    ) -> Union[Self, UserError]:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        # NOTE: The model can't handle multiple messages with images.
        # It throws "Invalid request 500" error.
        messages = messages[-1:]

        download_result = await download_inputs(
            file_storage, get_mime_types(IMAGE_TYPES), messages
        )

        usage_message = get_usage_message(get_file_exts(IMAGE_TYPES))

        if isinstance(download_result, str):
            return UserError(download_result, usage_message)

        image_count = sum(len(msg.image_inputs) for msg in download_result)
        if image_count == 0:
            return UserError("No image inputs were found", usage_message)

        if any(msg.has_empty_content() for msg in download_result):
            return UserError(
                "Messages with empty prompts are not allowed", usage_message
            )

        history = [res.to_content() for res in download_result]
        return cls(history=history[:-1], prompt=history[-1].parts)


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Only the last message will be taken into account.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
