"""
Classes to test the various models directly through the VertexAI SDK
"""

from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, assert_never

import vertexai
from vertexai.preview.generative_models import ChatSession as GenChatSession
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel
from vertexai.preview.language_models import ChatModel
from vertexai.preview.language_models import ChatSession as LangChatSession
from vertexai.preview.language_models import CodeChatModel
from vertexai.preview.language_models import (
    CodeChatSession as LangCodeChatSession,
)
from vertexai.preview.vision_models import (
    ImageGenerationModel,
    ImageGenerationResponse,
)

from aidial_adapter_vertexai.chat.gemini.adapter import default_safety_settings
from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithResources
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from client.chat.base import Chat
from client.utils.files import get_project_root
from client.utils.printing import print_info

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
        model = get_language_model_by_deployment(deployment)
        chat = model.start_chat()
        return cls(chat)

    async def send_message(
        self,
        tools: ToolsConfig,
        prompt: MessageWithResources,
        params: ModelParameters,
        usage: TokenUsage,
    ) -> AsyncIterator[str]:
        tools.not_supported()

        parameters = {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stop_sequences": params.stop,
            "top_p": params.top_p,
        }

        if not params.stream:
            parameters["candidate_count"] = params.n

        message = prompt.to_text()

        if params.stream:
            responses = self.chat.send_message_streaming(
                message=message, **parameters
            )
            for response in responses:
                yield response.text
        else:
            yield self.chat.send_message(message=message, **parameters).text


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=(
            [params.stop] if isinstance(params.stop, str) else params.stop
        ),
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
            case (
                ChatCompletionDeployment.GEMINI_PRO_1
                | ChatCompletionDeployment.GEMINI_PRO_VISION_1
                | ChatCompletionDeployment.GEMINI_PRO_VISION_1_5
            ):
                model = GenerativeModel(deployment)
            case _:
                raise ValueError(f"Unsupported model: {deployment}")

        chat = GenChatSession(model=model, history=[])

        return cls(chat)

    async def send_message(
        self,
        tools: ToolsConfig,
        prompt: MessageWithResources,
        params: ModelParameters,
        usage: TokenUsage,
    ) -> AsyncIterator[str]:
        config = create_generation_config(params)
        content = prompt.to_parts()

        log.debug(f"request config: {json_dumps(config)}")
        log.debug(f"request content: {json_dumps_short(content)}")

        if params.stream:
            response = await self.chat._send_message_streaming_async(
                content=content,  # type: ignore
                generation_config=config,
                safety_settings=default_safety_settings,
                tools=tools.to_gemini_tools(),
            )

            async for chunk in response:
                log.debug(f"response chunk: {json_dumps(chunk)}")
                yield chunk.text
        else:
            response = await self.chat._send_message_async(
                content=content,  # type: ignore
                generation_config=config,
                safety_settings=default_safety_settings,
                tools=tools.to_gemini_tools(),
            )

            log.debug(f"response: {json_dumps(response)}")
            yield response.text


class SDKImagenChat(Chat):
    model: ImageGenerationModel

    def __init__(self, model):
        self.model = model

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKImagenChat":
        vertexai.init(project=project, location=location)
        model = ImageGenerationModel.from_pretrained(deployment.value)
        return cls(model)

    @staticmethod
    def get_filename(ext) -> Path:
        dir = get_project_root() / "~images"
        dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.now()
        filename = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename += ext

        return dir / filename

    async def send_message(
        self,
        tools: ToolsConfig,
        prompt: MessageWithResources,
        params: ModelParameters,
        usage: TokenUsage,
    ) -> AsyncIterator[str]:
        tools.not_supported()

        response: ImageGenerationResponse = self.model.generate_images(
            prompt.to_text(), number_of_images=1, seed=None
        )

        print_info(f"Response: {response}")

        if len(response.images) == 0:
            raise RuntimeError("Expected 1 image in response, but got none")

        filename = str(SDKImagenChat.get_filename(".png"))
        response[0].save(filename)
        yield f"Generated image: {filename}"


async def create_sdk_chat(
    location: str, project: str, deployment: ChatCompletionDeployment
) -> Chat:
    match deployment:
        case (
            ChatCompletionDeployment.CHAT_BISON_1
            | ChatCompletionDeployment.CHAT_BISON_2
            | ChatCompletionDeployment.CHAT_BISON_2_32K
            | ChatCompletionDeployment.CODECHAT_BISON_1
            | ChatCompletionDeployment.CODECHAT_BISON_2
            | ChatCompletionDeployment.CODECHAT_BISON_2_32K
        ):
            return await SDKLangChat.create(location, project, deployment)
        case (
            ChatCompletionDeployment.GEMINI_PRO_1
            | ChatCompletionDeployment.GEMINI_PRO_VISION_1
            | ChatCompletionDeployment.GEMINI_PRO_VISION_1_5
        ):
            return await SDKGenChat.create(location, project, deployment)
        case ChatCompletionDeployment.IMAGEN_005:
            return await SDKImagenChat.create(location, project, deployment)
        case _:
            assert_never(deployment)


def get_language_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> ChatModel | CodeChatModel:
    match deployment:
        case (
            ChatCompletionDeployment.CHAT_BISON_1
            | ChatCompletionDeployment.CHAT_BISON_2
            | ChatCompletionDeployment.CHAT_BISON_2_32K
        ):
            return ChatModel.from_pretrained(deployment)
        case (
            ChatCompletionDeployment.CODECHAT_BISON_1
            | ChatCompletionDeployment.CODECHAT_BISON_2
            | ChatCompletionDeployment.CODECHAT_BISON_2_32K
        ):
            return CodeChatModel.from_pretrained(deployment)
        case _:
            raise ValueError(f"Unsupported model: {deployment}")
