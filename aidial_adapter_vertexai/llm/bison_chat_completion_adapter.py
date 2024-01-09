import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from aidial_sdk.chat_completion import Message
from typing_extensions import override

from aidial_adapter_vertexai.llm.bison_history_trimming import (
    get_discarded_messages_count,
)
from aidial_adapter_vertexai.llm.bison_prompt import BisonPrompt
from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.vertex_ai import get_vertex_ai_chat
from aidial_adapter_vertexai.llm.vertex_ai_chat import (
    VertexAIAuthor,
    VertexAIChat,
    VertexAIMessage,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage


class BisonChatCompletionAdapter(ChatCompletionAdapter[BisonPrompt]):
    def __init__(self, model: VertexAIChat):
        self.model = model

    @abstractmethod
    def _create_instance(self, prompt: BisonPrompt) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_parameters(self, params: ModelParameters) -> Dict[str, Any]:
        pass

    @override
    async def parse_prompt(self, messages: List[Message]) -> BisonPrompt:
        return BisonPrompt.parse(messages)

    @override
    async def truncate_prompt(
        self, prompt: BisonPrompt, max_prompt_tokens: int
    ) -> Tuple[BisonPrompt, int]:
        if max_prompt_tokens is None:
            return prompt, 0

        discarded = await get_discarded_messages_count(
            self, prompt, max_prompt_tokens
        )

        return (
            BisonPrompt(
                context=prompt.context,
                messages=prompt.messages[discarded:],
            ),
            discarded,
        )

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: BisonPrompt
    ) -> None:
        content_task = self.model.predict(
            params.stream,
            consumer,
            self._create_instance(prompt),
            self._create_parameters(params),
        )

        if params.stream:
            # Token usage isn't reported for streaming requests.
            # Computing it manually
            prompt_tokens, content = await asyncio.gather(
                self.count_prompt_tokens(prompt), content_task
            )
            completion_tokens = await self.count_completion_tokens(content)

            await consumer.set_usage(
                TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        else:
            await content_task

    @override
    async def count_prompt_tokens(self, prompt: BisonPrompt) -> int:
        return await self.model.count_tokens(self._create_instance(prompt))

    @override
    async def count_completion_tokens(self, string: str) -> int:
        messages = [VertexAIMessage(author=VertexAIAuthor.USER, content=string)]
        return await self.model.count_tokens(
            self._create_instance(BisonPrompt(context=None, messages=messages))
        )

    @override
    @classmethod
    async def create(cls, model_id: str, project_id: str, location: str):
        model = get_vertex_ai_chat(model_id, project_id, location)
        return cls(model)
