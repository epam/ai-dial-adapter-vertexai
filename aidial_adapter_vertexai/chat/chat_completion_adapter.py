from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from aidial_sdk.chat_completion import Message

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.utils.not_implemented import not_implemented

P = TypeVar("P")


class ChatCompletionAdapter(ABC, Generic[P]):
    @abstractmethod
    async def parse_prompt(
        self, tools: ToolsConfig, messages: List[Message]
    ) -> P | UserError:
        pass

    @abstractmethod
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: P
    ) -> None:
        pass

    @not_implemented
    async def truncate_prompt(
        self, prompt: P, max_prompt_tokens: int
    ) -> TruncatedPrompt: ...

    @not_implemented
    async def count_prompt_tokens(self, prompt: P) -> int: ...

    @not_implemented
    async def count_completion_tokens(self, string: str) -> int: ...
