"""
Classes to test the various models supported by the DIAL adapter.
"""

from typing import AsyncGenerator, List, Optional

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import CollectConsumer
from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from client.chat.base import Chat
from client.utils.concurrency import str_callback_to_stream_generator


class AdapterChat(Chat):
    model: ChatCompletionAdapter
    history: List[Message]

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
        self.history.append(Message(role=Role.USER, content=prompt))

        consumer: Optional[CollectConsumer] = None

        async def task(on_content):
            nonlocal consumer
            consumer = CollectConsumer(on_content=on_content)
            prompt = await self.model.parse_prompt(self.history)
            await self.model.chat(params, consumer, prompt)

        async def on_content(chunk: str):
            return

        async for chunk in str_callback_to_stream_generator(task, on_content):
            yield chunk

        assert consumer is not None

        self.history.append(
            Message(role=Role.ASSISTANT, content=consumer.content)
        )

        usage.accumulate(consumer.usage)
