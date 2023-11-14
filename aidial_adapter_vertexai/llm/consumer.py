from abc import ABC, abstractmethod
from typing import Callable, Coroutine, Optional

from aidial_sdk.chat_completion import Choice

from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage


class Consumer(ABC):
    @abstractmethod
    async def append_content(self, content: str):
        pass

    @abstractmethod
    async def set_usage(self, usage: TokenUsage):
        pass


class ChoiceConsumer(Consumer):
    usage: TokenUsage
    choice: Choice

    def __init__(self, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()

    async def append_content(self, content: str):
        self.choice.append_content(content)

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage


ContentCallback = Callable[[str], Coroutine[None, str, None]]


class CollectConsumer(Consumer):
    usage: TokenUsage
    content: str
    on_content: Optional[ContentCallback]

    def __init__(self, on_content: Optional[ContentCallback] = None):
        self.usage = TokenUsage()
        self.content = ""
        self.on_content = on_content

    async def append_content(self, content: str):
        if self.on_content:
            await self.on_content(content)
        self.content += content

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage
