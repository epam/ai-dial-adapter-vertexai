from typing import List, Self, Union

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.inputs import (
    messages_to_gemini_conversation,
)
from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessors,
    exclusive_validator,
)
from aidial_adapter_vertexai.chat.gemini.processors import (
    get_image_processor,
    get_pdf_processor,
    get_plain_text_processor,
    get_video_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import get_attachments
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class Gemini_1_0_Pro_Vision_Prompt(GeminiPrompt):
    @classmethod
    async def parse(
        cls,
        file_storage: FileStorage | None,
        tools: ToolsConfig,
        messages: List[Message],
    ) -> Union[Self, UserError]:
        tools.not_supported()

        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        messages = truncate_messages(messages)

        exclusive = exclusive_validator()

        processors = AttachmentProcessors(
            processors=[
                get_plain_text_processor(),
                get_image_processor(16, exclusive("image")),
                get_pdf_processor(16, exclusive("pdf")),
                get_video_processor(1, exclusive("video")),
            ],
            file_storage=file_storage,
        )

        conversation = await messages_to_gemini_conversation(
            processors, tools, messages
        )

        def usage_message():
            return _get_usage_message(processors.get_file_exts())

        if error_message := processors.get_error_message():
            return UserError(error_message, usage_message())

        if processors.resource_count == 0:
            return UserError("No documents were found", usage_message())

        return cls(
            system_instruction=conversation.system_instruction,
            contents=conversation.contents,
            tools=tools,
        )


def truncate_messages(messages: List[Message]) -> List[Message]:
    """
    Only a single message is supported by Gemini 1.0 Pro Vision,
    when non-text parts are present in the request.

    Otherwise, the following error is thrown:
        400 Unable to submit request because it has more than
        one contents field but model gemini-pro-vision only supports one.
        Remove all but one contents and try again.

    Conversely, multi-turn chat with text-only content is supported.
    However, we disable it, because it's hard to make it clear to the user,
    when the model works in a single-turn mode and when in a multi-turn mode.
    Thus, we make the model single-turn altogether.
    """

    msg1 = messages[-1]

    # Supporting a typical use-case when a user asks
    # a question about assistant-generated attachments:
    #   [-2]: Assistant message with attachments
    #   [-1]: User message with a question about attachments
    #         without attachments of their own
    if len(messages) > 1:
        msg2 = messages[-2]
        attachments1 = get_attachments(msg1)
        attachments2 = get_attachments(msg2)
        if msg2.role == Role.ASSISTANT and attachments2 and not attachments1:
            msg1 = msg1.copy()
            msg1.custom_content = msg2.custom_content

    return [msg1]


def _get_usage_message(exts: List[str]) -> str:
    return f"""
The application answers queries about attached documents.
Attach documents and ask questions about them in the same message.

Only the last message will be taken into account.

Attachments from the second to last message will be added to
the last message given that the last message doesn't have any attachments on its own.

The images, PDFs and videos must not be mixed in the same message.

Supported document extensions: {', '.join(exts)}.

Examples of queries:
- "Describe the picture" for one image,
- "What is depicted in these images?", "Compare the images" for multiple images,
- "Summarize the document" for a PDF,
- "What is happening in the video?" for a video.
""".strip()
