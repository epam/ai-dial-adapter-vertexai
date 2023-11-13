from enum import Enum
from typing import Optional

from pydantic import BaseModel

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from client.utils.cli import select_enum, select_option


class ClientMode(str, Enum):
    ADAPTER = "Adapter"
    SDK = "VertexAI SDK"

    def get_model_id(self) -> str:
        return self.value


class Parameters(BaseModel):
    chat_client: ClientMode
    model_id: ChatCompletionDeployment
    streaming: bool
    max_tokens: Optional[int]
    temperature: float

    @classmethod
    def get_interactive(cls) -> "Parameters":
        chat_client = select_enum("Client", ClientMode)
        model_id = select_enum("Model", ChatCompletionDeployment)
        streaming = select_option("Streaming", [False, True])

        max_tokens_str = input("Max tokens [int|no limit]: ")
        max_tokens = int(max_tokens_str) if max_tokens_str else None

        temperature_str = input("Temperature [float|0.0]: ")
        temperature = float(temperature_str) if temperature_str else 0.0

        return cls(
            chat_client=chat_client,
            model_id=model_id,
            streaming=streaming,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def to_model_parameters(self) -> ModelParameters:
        return ModelParameters(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.streaming,
        )

    def __str__(self) -> str:
        streaming = "streaming" if self.streaming else "non-streaming"
        max_tokens = f"max_tokens={self.max_tokens}" if self.max_tokens else ""
        params = ",".join(param for param in [streaming, max_tokens] if param)
        return f"[{params}] {self.model_id.value} {self.chat_client.value}"
