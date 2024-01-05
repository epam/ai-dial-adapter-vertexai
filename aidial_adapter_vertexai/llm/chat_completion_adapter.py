from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.universal_api.request import ModelParameters

P = TypeVar("P")


class ChatCompletionAdapter(ABC, Generic[P]):
    @abstractmethod
    async def parse_prompt(self, messages: List[Message]) -> P:
        pass

    @abstractmethod
    async def truncate_prompt(
        self, prompt: P, max_prompt_tokens: int
    ) -> Tuple[P, int]:
        pass

    @abstractmethod
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: P
    ) -> None:
        pass

    @abstractmethod
    async def count_prompt_tokens(self, prompt: P) -> int:
        pass

    @abstractmethod
    async def count_completion_tokens(self, string: str) -> int:
        pass
