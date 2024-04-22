import base64
from typing import List, Optional, Union, assert_never

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.process_inputs import (
    MessageWithInputs,
    download_inputs,
)
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


class GeminiPrompt(BaseModel):
    history: List[Content]
    prompt: List[Part]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def parse_non_vision(cls, messages: List[Message]) -> "GeminiPrompt":
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        messages = accommodate_first_system_message(messages)

        msgs = [
            MessageWithInputs(message=message, image_inputs=[])
            for message in messages
        ]

        history = list(map(to_content, msgs))
        return cls(history=history[:-1], prompt=history[-1].parts)

    @classmethod
    async def parse_vision(
        cls,
        file_storage: Optional[FileStorage],
        messages: List[Message],
    ) -> Union["GeminiPrompt", UserError]:
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

        history = list(map(to_content, download_result))
        return cls(history=history[:-1], prompt=history[-1].parts)

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]


def accommodate_first_system_message(messages: List[Message]) -> List[Message]:
    if len(messages) == 0:
        return messages

    first_message: Message = messages[0]
    if first_message.role != Role.SYSTEM:
        return messages

    if len(messages) == 1:
        first_message = first_message.copy()
        first_message.role = Role.USER
        return [first_message]

    second_message = messages[1]
    if second_message.role != Role.USER:
        return messages

    if first_message.content is None or second_message.content is None:
        return messages

    content = first_message.content + "\n" + second_message.content
    return [Message(role=Role.USER, content=content)] + messages[2:]


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
                "System messages other than the first system message are not allowed"
            )
        case Role.USER:
            return ChatSession._USER_ROLE
        case Role.ASSISTANT:
            return ChatSession._MODEL_ROLE
        case Role.FUNCTION:
            raise ValidationError("Function messages are not supported")
        case Role.TOOL:
            raise ValidationError("Tool messages are not supported")
        case _:
            assert_never(role)


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Only the last message will be taken into account.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
