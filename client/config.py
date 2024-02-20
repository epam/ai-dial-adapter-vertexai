import argparse
from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from client.utils.cli import select_enum, select_option


class ClientMode(str, Enum):
    ADAPTER = "Adapter"
    SDK = "SDK"


def enum_values(enum: Type[Enum]) -> list[str]:
    return [e.value for e in enum]


class Config(BaseModel):
    mode: ClientMode
    model_id: ChatCompletionDeployment
    streaming: bool
    max_tokens: Optional[int]
    temperature: float

    @classmethod
    def get_interactive(cls) -> "Config":
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--mode",
            type=str,
            required=False,
            help=f"One of {enum_values(ClientMode)}",
        )
        parser.add_argument(
            "--model",
            type=str,
            required=False,
            help=f"One of the available models: {enum_values(ChatCompletionDeployment)}",
        )
        parser.add_argument(
            "--max_tokens",
            type=int,
            required=False,
            help="Max tokens",
        )
        parser.add_argument(
            "-t",
            type=float,
            required=False,
            help="Temperature",
        )
        parser.add_argument(
            "--streaming",
            required=False,
            action="store_true",
            help="Streaming mode",
        )

        args = parser.parse_args()

        if args.mode is not None:
            mode = ClientMode(args.mode)
        else:
            mode = select_enum("Mode", ClientMode)

        if args.model is not None:
            model_id = ChatCompletionDeployment(args.model)
        else:
            model_id = select_enum("Model", ChatCompletionDeployment)

        if args.streaming is not None:
            streaming = args.streaming
        else:
            streaming = select_option("Streaming", [False, True])

        if args.max_tokens is not None:
            max_tokens = args.max_tokens
        else:
            max_tokens_str = input("Max tokens [int|no limit]: ")
            max_tokens = int(max_tokens_str) if max_tokens_str else None

        if args.t is not None:
            temperature = args.t
        else:
            temperature_str = input("Temperature [float|0.0]: ")
            temperature = float(temperature_str) if temperature_str else 0.0

        return cls(
            mode=mode,
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
        return f"[{params}] {self.model_id.value} {self.mode.value}"
