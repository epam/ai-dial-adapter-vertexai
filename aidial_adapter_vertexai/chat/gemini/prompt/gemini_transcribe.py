from typing import Self

from aidial_sdk.chat_completion import Message, MessageContentTextPart

from aidial_adapter_vertexai.chat.conversation.converters import (
    messages_to_conversation,
)
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.gemini.conversation_factory import (
    ConversationFactoryGenAI,
)
from aidial_adapter_vertexai.chat.gemini.processors import (
    GeminiAttachmentProcessorsGenAI,
    get_audio_processor,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPromptGenAI
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import is_system_role
from aidial_adapter_vertexai.dial_api.storage import FileStorage


class GeminiTranscribePrompt(GeminiPromptGenAI):
    """
    Prompt for the Gemini Transcribe models,
    which accept audio only and support neither system instructions nor multi-turn chat.
    """

    @classmethod
    async def parse(
        cls,
        file_storage: FileStorage | None,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: list[Message],
    ) -> Self | UserError:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        if any(is_system_role(message.role) for message in messages):
            raise ValidationError(
                "System messages aren't supported by the transcription model"
            )

        # The model rejects multi-turn chat, so every turn is transcribed independently.
        messages = messages[-1:]

        conversation_factory = ConversationFactoryGenAI()
        processors = GeminiAttachmentProcessorsGenAI(
            conversation_factory=conversation_factory,
            processors=[get_audio_processor()],
            file_storage=file_storage,
        )

        usage_message = get_usage_message(processors.get_file_exts())

        if _has_text(messages[-1]):
            return UserError(
                "Transcription models don't support text input.", usage_message
            )

        conversation = await messages_to_conversation(
            conversation_factory, processors, tools, messages
        )

        if error_message := processors.get_error_message():
            return UserError(error_message, usage_message)

        has_audio = any(
            part.inline_data
            for content in conversation.messages.raw_list
            for part in content.parts or []
        )
        if not has_audio:
            return UserError(
                "Attach an audio file to transcribe", usage_message
            )

        return cls(
            conversation=conversation, tools=tools, static_tools=static_tools
        )


def _has_text(message: Message) -> bool:
    match message.content:
        case str(text):
            return bool(text.strip())
        case list(parts):
            return any(
                isinstance(part, MessageContentTextPart) and part.text.strip()
                for part in parts
            )
        case _:
            return False


def get_usage_message(exts: list[str]) -> str:
    return f"""
The application transcribes attached audio files.
Attach one or more audio files to a message without any text to get their transcription.
Each message is transcribed independently of the previous ones.

Supported audio extensions: {", ".join(exts)}.
""".strip()
