import logging
from enum import Enum
from typing import Any, Dict, List, TypedDict

from google.cloud.aiplatform import compat, initializer
from google.cloud.aiplatform._streaming_prediction import (
    tensor_to_value,
    value_to_tensor,
)
from google.cloud.aiplatform.utils import PredictionAsyncClientWithOverride
from google.cloud.aiplatform_v1 import PredictRequest, StreamingPredictRequest
from google.cloud.aiplatform_v1beta1 import CountTokensRequest
from google.protobuf import json_format

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.protobuf import message_to_string
from aidial_adapter_vertexai.utils.timer import Timer


class VertexAIAuthor(str, Enum):
    USER = "user"
    BOT = "bot"


class VertexAIMessage(TypedDict):
    author: VertexAIAuthor
    content: str


class VertexAIChat:
    def __init__(
        self,
        client: PredictionAsyncClientWithOverride,
        endpoint: str,
    ):
        self.client = client
        self.endpoint = endpoint

    async def predict(
        self,
        stream: bool,
        consumer: Consumer,
        instance: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> str:
        if stream:
            return await self._predict_streaming(consumer, instance, parameters)
        else:
            return await self._predict_non_streaming(
                consumer, instance, parameters
            )

    async def _predict_non_streaming(
        self,
        consumer: Consumer,
        instance: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> str:
        request = PredictRequest(endpoint=self.endpoint, parameters=parameters)  # type: ignore
        request.instances.append(instance)  # type: ignore

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"predict request:\n{message_to_string(request)}")

        timer = Timer()
        response = await self.client.predict(request)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"predict response [{timer}]:\n{message_to_string(response)}"
            )

        prediction = json_format.MessageToDict(response.predictions.pb[0])  # type: ignore
        token_metadata = json_format.MessageToDict(
            response.metadata.pb["tokenMetadata"]  # type: ignore
        )

        content = prediction["candidates"][0]["content"]
        prompt_tokens = token_metadata["inputTokenCount"]["totalTokens"]
        completion_tokens = token_metadata["outputTokenCount"]["totalTokens"]

        await consumer.append_content(content)
        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

        return content

    async def _predict_streaming(
        self,
        consumer: Consumer,
        instance: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> str:
        request = StreamingPredictRequest(
            endpoint=self.endpoint,
            parameters=value_to_tensor(parameters),
        )
        request.inputs.append(value_to_tensor(instance))  # type: ignore

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"predict request:\n{message_to_string(request)}")

        timer = Timer()
        response = await self.client.server_streaming_predict(request)

        content = ""
        async for chunk in response:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"predict response chunk:\n{message_to_string(chunk)}"
                )

            outputs: List[Any] = [
                tensor_to_value(tensor._pb) for tensor in chunk.outputs
            ]

            content_chunk = outputs[0]["candidates"][0]["content"]
            await consumer.append_content(content_chunk)
            content += content_chunk

        log.debug(f"predict response finished [{timer}]")

        return content

    async def count_tokens(self, instance: Dict[str, Any]) -> int:
        request = CountTokensRequest(endpoint=self.endpoint)  # type: ignore
        request.instances.append(instance)  # type: ignore

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"count_token request:\n{message_to_string(request)}")

        timer = Timer()
        response = await self.client.select_version(
            compat.V1BETA1
        ).count_tokens(request)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"count_token response [{timer}]:\n{message_to_string(response)}"
            )

        return response.total_tokens

    @classmethod
    def create(cls, model: str, project: str, location: str):
        endpoint = f"projects/{project}/locations/{location}/publishers/google/models/{model}"

        client = initializer.global_config.create_client(
            # From the API code it looks the class is expected to implement ClientWithOverride
            client_class=PredictionAsyncClientWithOverride,  # type: ignore
        )

        return cls(client, endpoint)
