from typing import List, Self

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithInputs
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt


class GeminiProOnePrompt(GeminiPrompt):
    @classmethod
    def parse(cls, messages: List[Message]) -> Self:
        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        messages = accommodate_first_system_message(messages)

        history = [
            MessageWithInputs(message=message, inputs=[]).to_content()
            for message in messages
        ]

        return cls(history=history[:-1], prompt=history[-1].parts)


def accommodate_first_system_message(messages: List[Message]) -> List[Message]:
    """
    Attach the first system message to a subsequent user message.

    NOTE: it's possible to pass `system_instruction` to `GenerativeModel` constructor,
    however `system_instruction` field isn't yet fully integrated into the VertexAI SDK.
    In particular, it's not exposed in `GenerativeModel.count_tokens_async` method:
    https://github.com/googleapis/python-aiplatform/issues/3631
    """

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
