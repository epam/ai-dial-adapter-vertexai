from abc import ABC, abstractmethod
from typing import AsyncIterator

from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithResources
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage


class Chat(ABC):
    @classmethod
    @abstractmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "Chat":
        pass

    @abstractmethod
    def send_message(
        self,
        tools: ToolsConfig,
        prompt: MessageWithResources,
        params: ModelParameters,
        usage: TokenUsage,
    ) -> AsyncIterator[str]:
        pass
