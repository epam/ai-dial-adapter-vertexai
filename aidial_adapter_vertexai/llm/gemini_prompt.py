import base64
from typing import List, Optional, Union, assert_never

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.llm.exceptions import UserError, ValidationError
from aidial_adapter_vertexai.llm.process_inputs import (
    MessageWithInputs,
    download_inputs,
)
from aidial_adapter_vertexai.universal_api.storage import FileStorage

# Gemini Pro Pricing
# https://cloud.google.com/vertex-ai/pricing
# Image      : $0.0025  / image
# Video      : $0.002   / second
# Text Input : $0.00025 / 1k characters
# Text Output: $0.0005  / 1k characters

# Supported image types
# https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-multimodal-prompts?authuser=1#image-requirements
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png"]
TOKENS_PER_IMAGE = 258  # Contradicts Gemini Pro Pricing

# NOTE: see also supported video formats
# https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-multimodal-prompts?authuser=1#video-requirements


class GeminiPrompt(BaseModel):
    history: List[Content]
    prompt: List[Part]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    async def parse(
        cls,
        file_storage: Optional[FileStorage],
        download_images: bool,
        messages: List[Message],
    ) -> Union["GeminiPrompt", UserError]:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        image_types = SUPPORTED_IMAGE_TYPES if download_images else []
        res = await download_inputs(file_storage, image_types, messages)

        if isinstance(res, str):
            return UserError(res, get_usage(SUPPORTED_FILE_EXTS))
        else:
            history = list(map(to_content, res))
            return cls(history=history[:-1], prompt=history[-1].parts)

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]


def to_content(msg: MessageWithInputs) -> Content:
    message = msg.message
    content = message.content
    if content is None:
        raise ValidationError("Message content must be present")

    parts: List[Part] = []

    for image in msg.image_inputs:
        data = base64.b64decode(image.data, validate=True)
        parts.append(Part.from_data(data=data, mime_type=image.type))

    parts.append(Part.from_text(content))

    return Content(role=get_part_role(message.role), parts=parts)


def get_part_role(role: Role) -> str:
    match role:
        case Role.SYSTEM:
            raise ValidationError(
                "System messages are not allowed in Gemini models"
            )
        case Role.USER:
            return ChatSession._USER_ROLE
        case Role.ASSISTANT:
            return ChatSession._MODEL_ROLE
        case Role.FUNCTION:
            raise ValidationError("Function messages are not supported")
        case _:
            assert_never(role)


def get_usage(supported_exts: List[str]) -> str:
    return f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
