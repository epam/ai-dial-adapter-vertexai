from typing import AsyncIterator, assert_never

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    BisonChatModel,
    BisonChatSession,
)
from aidial_adapter_vertexai.llm.vertex_ai import (
    get_chat_model,
    get_code_chat_model,
    init_vertex_ai,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.timer import Timer
from client.chat.base import Chat
from client.utils.printing import print_info


async def get_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> BisonChatModel:
    model_id = deployment.get_model_id()
    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            return await get_chat_model(model_id)
        case ChatCompletionDeployment.CHAT_BISON_2:
            return await get_chat_model(model_id)
        case ChatCompletionDeployment.CHAT_BISON_2_32K:
            return await get_chat_model(model_id)
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            return await get_code_chat_model(model_id)
        case ChatCompletionDeployment.CODECHAT_BISON_2:
            return await get_code_chat_model(model_id)
        case ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return await get_code_chat_model(model_id)
        case _:
            assert_never(deployment)


class SDKBisonChat(Chat):
    chat: BisonChatSession
    model: BisonChatModel

    def __init__(self, model: BisonChatModel):
        self.model = model
        self.chat = self.model.start_chat()

    @classmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "SDKBisonChat":
        await init_vertex_ai(project, location)
        return cls(await get_model_by_deployment(deployment))

    @staticmethod
    def prepare_parameters(params: ModelParameters) -> dict:
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

        return parameters

    async def send_message_async(
        self, prompt: str, is_stream: bool, parameters: dict
    ) -> AsyncIterator[str]:
        if is_stream:
            stream = self.chat.send_message_streaming_async(
                message=prompt, **parameters
            )
            async for chunk in stream:
                yield chunk.text
        else:
            response = await self.chat.send_message_async(
                message=prompt, **parameters
            )
            yield response.text

    async def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncIterator[str]:
        with Timer("Timing (prompt count_tokens): {time}", print_info):
            prompt_tokens = self.chat.count_tokens(message=prompt).total_tokens

        with Timer("Timing (predict): {time}", print_info):
            parameters = self.prepare_parameters(params)

            completion = ""
            async for chunk in self.send_message_async(
                prompt, params.stream, parameters
            ):
                completion += chunk
                yield chunk

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
