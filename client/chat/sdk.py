"""
Classes to test the various models directly through the VertexAI SDK
"""

import logging
from typing import AsyncIterator

import vertexai
from vertexai.preview.generative_models import ChatSession as GenChatSession
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel
from vertexai.preview.language_models import ChatModel
from vertexai.preview.language_models import ChatSession as LangChatSession
from vertexai.preview.language_models import CodeChatModel
from vertexai.preview.language_models import (
    CodeChatSession as LangCodeChatSession,
)

from aidial_adapter_vertexai.llm.gemini_chat_completion_adapter import (
    BLOCK_NONE_SAFETY_SETTINGS,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from client.chat.base import Chat

log = logging.getLogger(__name__)

LangSession = LangChatSession | LangCodeChatSession


class SDKLangChat(Chat):
    chat: LangSession

    def __init__(self, chat: LangSession):
        self.chat = chat

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKLangChat":
        vertexai.init(project=project, location=location)
        model = get_model_by_deployment(deployment)
        chat = model.start_chat()
        return cls(chat)

    async def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncIterator[str]:
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


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=[params.stop]
        if isinstance(params.stop, str)
        else params.stop,
        top_p=params.top_p,
        candidate_count=1 if params.stream else params.n,
    )


class SDKGenChat(Chat):
    chat: GenChatSession

    def __init__(self, chat: GenChatSession):
        self.chat = chat

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKGenChat":
        vertexai.init(project=project, location=location)

        match deployment:
            case ChatCompletionDeployment.GEMINI_PRO_1:
                model = GenerativeModel(deployment)
            case _:
                raise ValueError(f"Unsupported model: {deployment}")

        chat = GenChatSession(model=model, history=[], raise_on_blocked=False)

        return cls(chat)

    async def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncIterator[str]:
        parameters = create_generation_config(params)

        if params.stream:
            response = await self.chat._send_message_streaming_async(
                content=prompt,
                generation_config=parameters,
                safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                tools=None,
            )

            async for chunk in response:
                yield chunk.text
        else:
            response = await self.chat._send_message_async(
                content=prompt,
                generation_config=parameters,
                safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                tools=None,
            )

            yield response.text


async def create_sdk_chat(
    location: str, project: str, deployment: ChatCompletionDeployment
) -> Chat:
    match deployment:
        case ChatCompletionDeployment.GEMINI_PRO_1:
            return await SDKGenChat.create(location, project, deployment)
        case _:
            return await SDKLangChat.create(location, project, deployment)


def get_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> ChatModel | CodeChatModel:
    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1 | ChatCompletionDeployment.CHAT_BISON_2 | ChatCompletionDeployment.CHAT_BISON_2_32K:
            return ChatModel.from_pretrained(deployment)
        case ChatCompletionDeployment.CODECHAT_BISON_1 | ChatCompletionDeployment.CODECHAT_BISON_2 | ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return CodeChatModel.from_pretrained(deployment)
        case _:
            raise ValueError(f"Unsupported model: {deployment}")
