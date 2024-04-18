from typing import List, Optional, Self, Union

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessor,
    InitValidator,
    exclusive_validator,
    max_count_validator,
    max_pdf_page_count_validator,
    process_messages,
    seq_validators,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.dial_api.storage import FileStorage

# Gemini capabilities: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts
# Prompt design: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing


# Tokens per image: 258. count_tokens API call takes this into account.
def get_image_processor(
    init_validator: InitValidator | None,
) -> AttachmentProcessor:
    # Validators maintain state, so we need to create a new instance each time.
    return AttachmentProcessor(
        file_types={
            "image/jpeg": ["jpg", "jpeg"],
            "image/png": "png",
        },
        init_validator=seq_validators(init_validator, max_count_validator(16)),
    )


# The maximum file size for a PDF is 50MB. Currently not checked.
# PDFs are treated as images, so a single page of a PDF is treated as one image.
def get_pdf_processor(
    init_validator: InitValidator | None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={"application/pdf": "pdf"},
        init_validator=init_validator,
        post_validator=max_pdf_page_count_validator(16),
    )


# Audio in the video is ignored.
# Videos are sampled at 1fps. Each video frame accounts for 258 tokens.
# The video is automatically truncated to the first two minutes.
def get_video_processor(
    init_validator: InitValidator | None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={
            "video/mp4": "mp4",
            "video/mov": "mov",
            "video/mpeg": "mpeg",
            "video/mpg": "mpg",
            "video/avi": "avi",
            "video/wmv": "wmv",
            "video/mpegps": "mpegps",
            "video/flv": "flv",
        },
        init_validator=seq_validators(init_validator, max_count_validator(1)),
    )


def get_file_exts(processors: List[AttachmentProcessor]) -> List[str]:
    return [ext for proc in processors for ext in proc.file_exts]


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
        # So we feed to the model only the last message,
        # which essentially turns it into a text completion model.
        messages = messages[-1:]

        exclusive = exclusive_validator()

        processors = [
            get_image_processor(exclusive("image")),
            get_pdf_processor(exclusive("pdf")),
            get_video_processor(exclusive("video")),
        ]

        download_result = await process_messages(
            processors, file_storage, messages
        )

        usage_message = get_usage_message(get_file_exts(processors))

        if isinstance(download_result, str):
            return UserError(download_result, usage_message)

        input_count = sum(len(msg.inputs) for msg in download_result)
        if input_count == 0:
            return UserError("No inputs were found", usage_message)

        if any(msg.has_empty_content() for msg in download_result):
            return UserError(
                "Messages with empty prompts are not allowed", usage_message
            )

        history = [res.to_content() for res in download_result]
        return cls(history=history[:-1], prompt=history[-1].parts)


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
### Usage

The application answers queries about attached documents.
Attach documents and ask questions about them in the same message.

Only the last message will be taken into account.

Supported document types: {', '.join(supported_exts)}.

The images, PDFs and videos must not be mixed in the same message.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
