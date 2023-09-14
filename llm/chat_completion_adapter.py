from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from vertexai.language_models._language_models import ChatMessage

from llm.exception import ValidationError
from universal_api.token_usage import TokenUsage


def to_chat_message(message: BaseMessage) -> ChatMessage:
    author = "bot" if isinstance(message, AIMessage) else "user"
    return ChatMessage(author=author, content=message.content)


ChatCompletionResponse = Tuple[str, TokenUsage]


class ChatCompletionAdapter(ABC):
    @abstractmethod
    async def _call(
        self,
        streaming: bool,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> ChatCompletionResponse:
        pass

    async def completion(
        self, streaming: bool, prompt: str
    ) -> ChatCompletionResponse:
        return await self._call(streaming, None, [], prompt)

    async def chat(
        self, streaming: bool, history: List[BaseMessage]
    ) -> ChatCompletionResponse:
        messages = history.copy()

        if len(messages) == 0:
            raise ValidationError(
                "The chat history must have at least one message"
            )

        context: Optional[str] = None
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            context = messages.pop(0).content
            context = context if context.strip() else None

        if len(messages) == 0 and context is not None:
            raise ValidationError(
                "The chat history must have at least one non-system message"
            )

        role: Optional[str] = None
        for message in messages:
            if isinstance(message, SystemMessage):
                raise ValidationError(
                    "System messages other than the initial system message are not allowed"
                )
            if message.content == "":
                raise ValidationError("Empty messages are not allowed")

            if role is not None and role == message.type:
                raise ValidationError("Messages must alternate between authors")

            role = message.type

        message_history = list(map(to_chat_message, messages[:-1]))

        if len(message_history) % 2 != 0:
            raise ValidationError(
                "There should be odd number of messages for correct alternating turn"
            )

        prompt_message = messages[-1]

        if not isinstance(prompt_message, HumanMessage):
            raise ValidationError("The last message must be a user message")

        return await self._call(
            streaming, context, message_history, prompt_message.content
        )
