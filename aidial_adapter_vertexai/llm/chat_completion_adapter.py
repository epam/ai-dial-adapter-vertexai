from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator, List, Optional, Tuple

from vertexai.preview.language_models import (
    ChatMessage,
    ChatModel,
    ChatSession,
    CodeChatModel,
    CountTokensResponse,
)

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer


class ChatAuthor(str, Enum):
    USER = ChatSession.USER_AUTHOR
    BOT = ChatSession.MODEL_AUTHOR

    def __repr__(self) -> str:
        return f"{self.value!r}"


BisonChatModel = ChatModel | CodeChatModel


class ChatCompletionAdapter(ABC):
    model: BisonChatModel

    def __init__(self, model: BisonChatModel):
        self.model = model

    @classmethod
    @abstractmethod
    async def create(
        cls, model_id: str, project_id: str, location: str
    ) -> "ChatCompletionAdapter":
        pass

    @abstractmethod
    def send_message_async(
        self,
        params: ModelParameters,
        context: Optional[str],
        messages: List[ChatMessage],
        prompt: str,
    ) -> AsyncIterator[str]:
        pass

    async def chat(
        self,
        consumer: Consumer,
        context: Optional[str],
        messages: List[ChatMessage],
        params: ModelParameters,
    ) -> None:
        prompt_tokens = await self.count_prompt_tokens(context, messages)
        message_history, prompt = _parse_messages(messages)

        with Timer("predict timing: {time}", log.debug):
            log.debug(
                "predict request: "
                f"parameters=({params}), "
                f"context={context!r}, "
                f"history={messages}, "
                f"prompt={prompt!r}"
            )

            completion = ""

            async for chunk in self.send_message_async(
                params, context, message_history, prompt
            ):
                completion += chunk
                await consumer.append_content(chunk)

            log.debug(f"predict response: {completion!r}")

        await consumer.append_content(None)

        completion_tokens = await self.count_completion_tokens(completion)

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    async def count_prompt_tokens(
        self, context: Optional[str], messages: List[ChatMessage]
    ) -> int:
        message_history, prompt = _parse_messages(messages)
        chat_session = self.model.start_chat(
            context=context, message_history=message_history
        )

        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = chat_session.count_tokens(message=prompt)
            log.debug(
                f"count_tokens[prompt] response: {_display_token_count(resp)}"
            )
            return resp.total_tokens

    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = self.model.start_chat().count_tokens(message=string)
            log.debug(
                f"count_tokens[completion] response: {_display_token_count(resp)}"
            )
            return resp.total_tokens


def _parse_messages(
    messages: List[ChatMessage],
) -> Tuple[List[ChatMessage], str]:
    if len(messages) == 0:
        raise ValidationError("Messages should not be empty")

    message_history = messages[:-1]
    prompt = messages[-1].content

    return message_history, prompt


def _display_token_count(response: CountTokensResponse) -> str:
    return f"tokens: {response.total_tokens}, billable characters: {response.total_billable_characters}"
