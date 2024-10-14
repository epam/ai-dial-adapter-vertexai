from typing import AsyncIterator, List, Optional, TypedDict

from typing_extensions import override
from vertexai.preview.language_models import ChatModel, CodeChatModel

from aidial_adapter_vertexai.chat.bison.base import BisonChatCompletionAdapter
from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.vertex_ai import (
    get_chat_model,
    get_code_chat_model,
)


class CodeChatParamsBase(TypedDict, total=False):
    max_output_tokens: Optional[int]
    temperature: Optional[float]


class ChatParamsBase(TypedDict, total=False):
    max_output_tokens: Optional[int]
    temperature: Optional[float]

    # Extra compared to CodeChatParams
    stop_sequences: Optional[List[str]]
    top_k: Optional[int]
    top_p: Optional[float]


ChatParamsStream = ChatParamsBase
CodeChatParamsStream = CodeChatParamsBase


class NoStreamParams(TypedDict, total=False):
    candidate_count: Optional[int]


class ChatParamsNoStream(ChatParamsBase, NoStreamParams):
    pass


class CodeChatParamsNoStream(CodeChatParamsBase, NoStreamParams):
    pass


class BisonChatAdapter(BisonChatCompletionAdapter):
    model: ChatModel

    @classmethod
    async def create(cls, model_id: str) -> "BisonChatAdapter":
        return cls(await get_chat_model(model_id))

    def prepare_parameters_no_stream(
        self, params: ModelParameters
    ) -> ChatParamsNoStream:
        return {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stop_sequences": params.stop,
            "top_p": params.top_p,
            "candidate_count": params.n,
        }

    def prepare_parameters_stream(
        self, params: ModelParameters
    ) -> ChatParamsStream:
        return {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stop_sequences": params.stop,
            "top_p": params.top_p,
        }

    @override
    async def send_message_async(
        self, params: ModelParameters, prompt: BisonPrompt
    ) -> AsyncIterator[str]:
        chat = self.model.start_chat(
            context=prompt.context, message_history=prompt.history
        )

        generic_validate_parameters(params)

        if params.stream:
            stream = chat.send_message_streaming_async(
                message=prompt.user_prompt,
                **self.prepare_parameters_stream(params),
            )
            async for chunk in stream:
                yield chunk.text
        else:
            response = await chat.send_message_async(
                message=prompt.user_prompt,
                **self.prepare_parameters_no_stream(params),
            )
            yield response.text


class BisonCodeChatAdapter(BisonChatCompletionAdapter):
    model: CodeChatModel

    @classmethod
    async def create(cls, model_id: str) -> "BisonCodeChatAdapter":
        return cls(await get_code_chat_model(model_id))

    def validate_parameters(self, params: ModelParameters) -> None:
        if params.stop is not None and params.stop != []:
            raise ValidationError(
                "stop sequences are not supported for code chat model"
            )

        if params.top_p is not None:
            raise ValidationError("top_p is not supported for code chat model")

    def prepare_parameters_no_stream(
        self, params: ModelParameters
    ) -> CodeChatParamsNoStream:
        return {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
            "candidate_count": params.n,
        }

    def prepare_parameters_stream(
        self, params: ModelParameters
    ) -> CodeChatParamsStream:
        return {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

    @override
    async def send_message_async(
        self, params: ModelParameters, prompt: BisonPrompt
    ) -> AsyncIterator[str]:
        chat = self.model.start_chat(
            context=prompt.context, message_history=prompt.history
        )

        generic_validate_parameters(params)
        self.validate_parameters(params)

        if params.stream:
            stream = chat.send_message_streaming_async(
                message=prompt.user_prompt,
                **self.prepare_parameters_stream(params),
            )
            async for chunk in stream:
                yield chunk.text
        else:
            response = await chat.send_message_async(
                message=prompt.user_prompt,
                **self.prepare_parameters_no_stream(params),
            )
            yield response.text


def generic_validate_parameters(params: ModelParameters) -> None:
    # Currently n>1 is emulated by calling the model n times.
    # So the individual generation requests are expected to have n=1 or unset.
    if params.n is not None and params.n > 1:
        raise ValueError("n is expected to be 1 or unset")
