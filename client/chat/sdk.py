from typing import AsyncIterator, assert_never

import vertexai
from vertexai.preview.language_models import ChatModel
from vertexai.preview.language_models import ChatSession as LangChatSession
from vertexai.preview.language_models import CodeChatModel
from vertexai.preview.language_models import (
    CodeChatSession as LangCodeChatSession,
)

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.timer import Timer
from client.chat.base import Chat
from client.utils.printing import print_info

LangSession = LangChatSession | LangCodeChatSession
LangModel = ChatModel | CodeChatModel


def get_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> LangModel:
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


class SDKLangChat(Chat):
    chat: LangSession
    model: LangModel

    def __init__(self, model: LangModel):
        self.model = model
        self.chat = self.model.start_chat()

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKLangChat":
        vertexai.init(project=project, location=location)
        return cls(get_model_by_deployment(deployment))

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

        with Timer("Timing (prompt count_tokens): {time}", print_info):
            prompt_tokens = self.chat.count_tokens(message=prompt).total_tokens

        with Timer("Timing (predict): {time}", print_info):
            completion = ""

            if params.stream:
                stream = self.chat.send_message_streaming_async(
                    message=prompt, **parameters
                )
                async for chunk in stream:
                    completion += chunk.text
                    yield chunk.text
            else:
                response = await self.chat.send_message_async(
                    message=prompt, **parameters
                )
                completion += response.text
                yield response.text

            print("")

        with Timer("Timing (completion count_tokens): {time}", print_info):
            completion_tokens = (
                self.model.start_chat()
                .count_tokens(message=completion)
                .total_tokens
            )

        usage.accumulate(
            TokenUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        )
