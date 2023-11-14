import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.vertex_ai import get_vertex_ai_chat
from aidial_adapter_vertexai.llm.vertex_ai_chat import (
    VertexAIAuthor,
    VertexAIChat,
    VertexAIMessage,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage


class ChatCompletionAdapter(ABC):
    def __init__(self, model: VertexAIChat):
        self.model = model

    @abstractmethod
    def _create_instance(
        self, context: Optional[str], messages: List[VertexAIMessage]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_parameters(self, params: ModelParameters) -> Dict[str, Any]:
        pass

    async def chat(
        self,
        consumer: Consumer,
        context: Optional[str],
        messages: List[VertexAIMessage],
        params: ModelParameters,
    ) -> None:
        content_task = self.model.predict(
            params.stream,
            consumer,
            self._create_instance(context, messages),
            self._create_parameters(params),
        )

        if params.stream:
            # Token usage isn't reported for streaming requests.
            # Computing it manually
            prompt_tokens, content = await asyncio.gather(
                self.count_prompt_tokens(context, messages), content_task
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

    async def count_prompt_tokens(
        self, context: Optional[str], messages: List[VertexAIMessage]
    ) -> int:
        return await self.model.count_tokens(
            self._create_instance(context, messages)
        )

    async def count_completion_tokens(self, string: str) -> int:
        return await self.model.count_tokens(
            self._create_instance(
                None,
                [VertexAIMessage(author=VertexAIAuthor.USER, content=string)],
            )
        )

    @classmethod
    def create(cls, model_id: str, project_id: str, location: str):
        model = get_vertex_ai_chat(model_id, project_id, location)
        return cls(model)
