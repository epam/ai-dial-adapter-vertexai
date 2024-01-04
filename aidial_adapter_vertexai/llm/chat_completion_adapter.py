from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, assert_never

from vertexai.preview.language_models import (
    ChatMessage,
    ChatModel,
    ChatSession,
    CodeChatModel,
    CodeChatSession,
    CountTokensResponse,
)

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.llm.vertex_ai import get_vertex_ai_chat
from aidial_adapter_vertexai.llm.vertex_ai_chat import VertexAIChat
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer


class ChatAuthor(str, Enum):
    USER = CodeChatSession.USER_AUTHOR
    BOT = CodeChatSession.MODEL_AUTHOR


BisonChatModel = ChatModel | CodeChatModel
BisonChatSession = ChatSession | CodeChatSession


def get_model_by_deployment(
    deployment: ChatCompletionDeployment,
) -> BisonChatModel:
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


def display_token_count(response: CountTokensResponse) -> str:
    return f"tokens: {response.total_tokens}, billable characters: {response.total_billable_characters}"


class ChatCompletionAdapter(ABC):
    model: VertexAIChat
    lang_model: BisonChatModel

    def __init__(self, model: VertexAIChat, lang_model: BisonChatModel):
        self.model = model
        self.lang_model = lang_model

    @abstractmethod
    def _create_instance(
        self, context: Optional[str], messages: List[ChatMessage]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_parameters(self, params: ModelParameters) -> Dict[str, Any]:
        pass

    def prepare_parameters(self, params: ModelParameters) -> dict:
        parameters = {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        stop_sequences = (
            [params.stop] if isinstance(params.stop, str) else params.stop
        )

        is_codechat = isinstance(self.lang_model, CodeChatModel)

        if stop_sequences is not None:
            if is_codechat:
                raise ValidationError(
                    "stop sequences are not supported for code chat model"
                )
            parameters["stop_sequences"] = stop_sequences

        if params.top_p is not None:
            if is_codechat:
                raise ValidationError(
                    "top_p is not supported for code chat model"
                )
            parameters["top_p"] = params.top_p

        if params.stream:
            if params.n is not None and params.n > 1:
                raise ValidationError("n>1 is not supported in streaming mode")
        else:
            parameters["candidate_count"] = params.n

        return parameters

    @staticmethod
    async def send_message_async(
        chat: BisonChatSession, prompt: str, is_stream: bool, parameters: dict
    ) -> AsyncIterator[str]:
        if is_stream:
            stream = chat.send_message_streaming_async(
                message=prompt, **parameters
            )
            async for chunk in stream:
                yield chunk.text
        else:
            response = await chat.send_message_async(
                message=prompt, **parameters
            )
            yield response.text

    async def chat(
        self,
        consumer: Consumer,
        context: Optional[str],
        messages: List[ChatMessage],
        params: ModelParameters,
    ) -> None:
        # TODO: make sure `context` is supported by CodeChatModel

        if len(messages) == 0:
            raise ValidationError("Messages should not be empty")
        last_message = messages[-1]

        if last_message.author != ChatAuthor.USER:
            raise ValidationError(
                "The last message should be a message from user"
            )

        message_history = messages[:-1]
        prompt = last_message.content

        chat_session = self.lang_model.start_chat(
            context=context, message_history=message_history
        )

        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = chat_session.count_tokens(message=prompt)
            log.debug(
                f"count_tokens[prompt] response: {display_token_count(resp)}"
            )
            prompt_tokens = resp.total_tokens

        with Timer("predict timing: {time}", log.debug):
            parameters = self.prepare_parameters(params)

            log.debug(
                "predict request: "
                f"parameters=({parameters}), "
                f"context={chat_session._context!r}, "
                f"history={chat_session.message_history}, "
                f"prompt={prompt!r}"
            )

            completion = ""
            async for chunk in self.send_message_async(
                chat_session, prompt, params.stream, parameters
            ):
                completion += chunk
                await consumer.append_content(chunk)

            log.debug(f"predict response: {completion!r}")

        await consumer.append_content(None)

        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = self.lang_model.start_chat().count_tokens(message=completion)
            log.debug(
                f"count_tokens[completion] response: {display_token_count(resp)}"
            )
            completion_tokens = resp.total_tokens

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    async def count_prompt_tokens(
        self, context: Optional[str], messages: List[ChatMessage]
    ) -> int:
        return await self.model.count_tokens(
            self._create_instance(context, messages)
        )

    async def count_completion_tokens(self, string: str) -> int:
        return await self.model.count_tokens(
            self._create_instance(
                None,
                [ChatMessage(author=ChatAuthor.USER, content=string)],
            )
        )

    @classmethod
    def create(
        cls,
        lang_model: BisonChatModel,
        model_id: str,
        project_id: str,
        location: str,
    ):
        model = get_vertex_ai_chat(model_id, project_id, location)
        return cls(model, lang_model)
