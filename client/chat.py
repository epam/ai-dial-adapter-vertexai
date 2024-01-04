import logging
from abc import ABC, abstractmethod
from typing import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Dict,
    List,
    Optional,
    Union,
    assert_never,
    cast,
)

import vertexai
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
from vertexai.preview.generative_models import ChatSession as GenChatSession
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
)
from vertexai.preview.language_models import ChatModel
from vertexai.preview.language_models import ChatSession as LangChatSession
from vertexai.preview.language_models import CodeChatModel
from vertexai.preview.language_models import (
    CodeChatSession as LangCodeChatSession,
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
from aidial_adapter_vertexai.utils.protobuf import message_to_dict
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
    ) -> AsyncIterator[str]:
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
        case ChatCompletionDeployment.GEMINI_PRO_1:
            raise NotImplementedError("Gemini Pro is not supported yet")
        case _:
            assert_never(deployment)


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
        parameters: GenerationConfig = GenerationConfig(
            max_output_tokens=params.max_tokens,
            temperature=params.temperature,
            stop_sequences=[params.stop]
            if isinstance(params.stop, str)
            else params.stop,
            top_p=params.top_p,
            candidate_count=1 if params.stream else params.n,
        )

        HarmCategory = gapic_content_types.HarmCategory
        HarmBlockThreshold = (
            gapic_content_types.SafetySetting.HarmBlockThreshold
        )

        safety_settings: Dict[HarmCategory, HarmBlockThreshold] = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        response: Union[
            Awaitable[GenerationResponse],
            Awaitable[AsyncIterable[GenerationResponse]],
        ] = self.chat.send_message_async(
            content=prompt,
            generation_config=parameters,
            safety_settings=safety_settings,
            stream=params.stream,
            tools=None,
        )

        if params.stream:
            response = cast(
                Awaitable[AsyncIterable[GenerationResponse]], response
            )
            async for chunk in await response:
                # print(chunk)
                yield chunk.text
        else:
            response = cast(Awaitable[GenerationResponse], response)
            resp = await response
            print(resp)
            usage_proto = resp._raw_response.usage_metadata
            usage_dict = message_to_dict(usage_proto)
            print(usage_dict)
            yield resp.text


async def create_sdk_chat(
    location: str, project: str, model_id: ChatCompletionDeployment
) -> Chat:
    vertexai.init(project=project, location=location)

    match model_id:
        case ChatCompletionDeployment.GEMINI_PRO_1:
            return await SDKGenChat.create(location, project, model_id)
        case _:
            return await SDKLangChat.create(location, project, model_id)


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
