import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage

log = logging.getLogger(__name__)


class Chat(ABC):
    @classmethod
    @abstractmethod
    async def create(
        cls, location: str, project: str, deployment: ChatCompletionDeployment
    ) -> "Chat":
        pass

    @abstractmethod
    def send_message(
        self, prompt: str, params: ModelParameters, usage: TokenUsage
    ) -> AsyncIterator[str]:
        pass
