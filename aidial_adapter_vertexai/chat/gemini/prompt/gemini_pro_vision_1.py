from typing import List, Optional, Self, Union

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.process_inputs import download_inputs
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.dial_api.storage import FileStorage

# Pricing info: https://cloud.google.com/vertex-ai/pricing
# Supported image types:
# https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-multimodal-prompts?authuser=1#image-requirements
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png"]
# NOTE: Tokens per image: 258. count_tokens API call takes this into account.
# Up to 16 images. Total max size 4MB.

# NOTE: See also supported video formats:
# https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-multimodal-prompts?authuser=1#video-requirements
# Tokens per video: 1032


class GeminiProOneVisionPrompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls, file_storage: Optional[FileStorage], messages: List[Message]
    ) -> Union[Self, UserError]:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        # NOTE: Vision model can't handle multiple messages with images.
        # It throws "Invalid request 500" error.
        messages = messages[-1:]

        download_result = await download_inputs(
            file_storage, SUPPORTED_IMAGE_TYPES, messages
        )

        usage_message = get_usage_message(SUPPORTED_FILE_EXTS)

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
