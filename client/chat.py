import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, assert_never

import vertexai
from vertexai.preview.language_models import (
    ChatModel,
    ChatSession,
    CodeChatModel,
    CodeChatSession,
)

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import CollectConsumer
from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_chat import (
    VertexAIAuthor,
    VertexAIMessage,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from client.utils.concurrency import str_callback_to_stream_generator

log = logging.getLogger(__name__)


class Chat(ABC):
    @classmethod
    @abstractmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "Chat":
        pass

    @abstractmethod
    def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncGenerator[str, None]:
        pass


def get_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> ChatModel | CodeChatModel:
    def get_chat():
        return ChatModel.from_pretrained(deployment)

    def get_codechat():
        return CodeChatModel.from_pretrained(deployment)

    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            return get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2:
            return get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2_32K:
            return get_chat()
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            return get_codechat()
        case ChatCompletionDeployment.CODECHAT_BISON_2:
            return get_codechat()
        case ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return get_codechat()
        case _:
            assert_never(deployment)


class SDKChat(Chat):
    chat: ChatSession | CodeChatSession

    def __init__(self, chat: ChatSession | CodeChatSession):
        self.chat = chat

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKChat":
        vertexai.init(project=project, location=location)
        model = get_model_by_deployment(deployment)
        chat = model.start_chat()
        return cls(chat)

    async def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncGenerator[str, None]:
        parameters = {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stop_sequences": [params.stop]
            if isinstance(params.stop, str)
            else params.stop,
            "top_p": params.top_p,
        }

        if not params.stream:
            parameters["candidate_count"] = params.n

        if params.stream:
            responses = self.chat.send_message_streaming(
                message=prompt, **parameters
            )
            for response in responses:
                yield response.text
        else:
            yield self.chat.send_message(message=prompt, **parameters).text


class AdapterChat(Chat):
    model: ChatCompletionAdapter
    history: List[VertexAIMessage]

    def __init__(self, model: ChatCompletionAdapter):
        self.model = model
        self.history = []

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "AdapterChat":
        model = await get_chat_completion_model(
            location=location, project_id=project, deployment=deployment
        )

        return cls(model)

    async def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncGenerator[str, None]:
        self.history.append(
            VertexAIMessage(author=VertexAIAuthor.USER, content=prompt)
        )

        consumer: Optional[CollectConsumer] = None

        async def task(on_content):
            nonlocal consumer
            consumer = CollectConsumer(on_content=on_content)
            await self.model.chat(consumer, None, self.history, params)

        async def on_content(chunk: str):
            return

        async for chunk in str_callback_to_stream_generator(task, on_content):
            yield chunk

        assert consumer is not None

        self.history.append(
            VertexAIMessage(author=VertexAIAuthor.BOT, content=consumer.content)
        )

        usage.accumulate(consumer.usage)
