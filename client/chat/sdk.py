import logging
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
from client.chat.base import Chat

log = logging.getLogger(__name__)


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
