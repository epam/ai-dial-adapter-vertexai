import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from logging import DEBUG
from typing import NotRequired

import pydantic
from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.exceptions import InternalServerError
from google.genai.client import Client as GenAIClient
from pydantic import BaseModel
from typing_extensions import TypedDict

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs,
)
from aidial_adapter_vertexai.dial_api.resource import AttachmentResource
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.types import (
    Embedding,
    make_embeddings_response,
    vector_to_embedding,
)
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.pydantic import ExtraAllowModel

# See the documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api

SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"]


class ImageInput(TypedDict):
    bytesBase64Encoded: str


class Instance(TypedDict):
    image: NotRequired[ImageInput]
    text: NotRequired[str]


class Parameters(TypedDict):
    dimension: int


class PredictRequest(TypedDict):
    instances: list[Instance]
    parameters: NotRequired[Parameters]


class Prediction(ExtraAllowModel):
    textEmbedding: list[float] | None = None
    imageEmbedding: list[float] | None = None


class PredictResponse(ExtraAllowModel):
    predictions: list[Prediction]


class ModelRequest(BaseModel):
    image: bytes | None = None
    contextual_text: str | None = None

    def count_input_tokens(self) -> int:
        # The model doesn't report the number of input tokens.
        # However, one could count it oneself:
        # https://cloud.google.com/vertex-ai/generative-ai/pricing#embedding-models
        # As of 29 Jul 2024, one image costs as much as 500 text input characters
        ret = len(self.contextual_text or "")
        if self.image:
            ret += 500
        return ret

    def to_instance(self) -> Instance:
        instance: Instance = {}
        if self.image:
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(self.image).decode()
            }
        if self.contextual_text:
            instance["text"] = self.contextual_text
        return instance

    def extract_embeddings(
        self, prediction: Prediction
    ) -> tuple[list[float], int]:
        vector: list[float] | None = None
        if self.image:
            vector = prediction.imageEmbedding
        else:
            vector = prediction.textEmbedding

        if vector is None:
            raise ValueError("No embeddings returned")

        return vector, self.count_input_tokens()


async def compute_embeddings(
    request: ModelRequest,
    client: GenAIClient,
    model_id: str,
    base64_encode: bool,
    dimensions: int | None,
) -> tuple[Embedding, int]:
    body: PredictRequest = {"instances": [request.to_instance()]}
    if dimensions:
        body["parameters"] = {"dimension": dimensions}

    if log.isEnabledFor(DEBUG):
        log.debug(f"request: {json_dumps_short(body)}")

    response = await client.aio._api_client.async_request(
        "post", f"publishers/google/models/{model_id}:predict", dict(body)
    )
    try:
        prediction = PredictResponse.model_validate_json(
            response.body or ""
        ).predictions[0]
    except (pydantic.ValidationError, IndexError) as e:
        log.error(f"Invalid upstream response: {e}")
        raise InternalServerError("Invalid upstream response") from e

    if log.isEnabledFor(DEBUG):
        log.debug(f"response: {json_dumps_short(prediction)}")

    vec, tokens = request.extract_embeddings(prediction)

    return vector_to_embedding(base64_encode, vec), tokens


def validate_request(request: EmbeddingsRequest) -> None:
    if request.custom_fields is not None:
        if request.custom_fields.instruction is not None:
            raise ValidationError("Instruction prompt is not supported")

        if request.custom_fields.type is not None:
            raise ValidationError(
                "The embedding model does not support embedding types"
            )


def _validate_content_type(content_type: str, supported_types: list[str]):
    if content_type not in supported_types:
        raise UserError(
            f"Unsupported attachment content type: {content_type}. "
            f"Supported attachment  types: {', '.join(supported_types)}."
        )


async def get_requests(
    storage: FileStorage | None, request: EmbeddingsRequest
) -> AsyncIterator[ModelRequest]:
    async def download_image(attachment: Attachment) -> bytes:
        resource = await AttachmentResource(attachment=attachment).download(
            storage
        )
        _validate_content_type(resource.type, SUPPORTED_IMAGE_TYPES)
        return resource.data

    async def on_text(text: str):
        return ModelRequest(contextual_text=text)

    async def on_attachment(attachment: Attachment):
        return ModelRequest(image=await download_image(attachment))

    async def on_mixed(inputs: list[str | Attachment]) -> ModelRequest:
        if len(inputs) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(inputs) == 1:
            if isinstance(inputs[0], str):
                return await on_text(inputs[0])
            else:
                return await on_attachment(inputs[0])
        elif len(inputs) == 2:
            if isinstance(inputs[0], str) and isinstance(inputs[1], Attachment):
                return ModelRequest(
                    contextual_text=inputs[0],
                    image=await download_image(inputs[1]),
                )
            elif isinstance(inputs[0], Attachment) and isinstance(
                inputs[1], str
            ):
                return ModelRequest(
                    contextual_text=inputs[1],
                    image=await download_image(inputs[0]),
                )
            else:
                raise ValidationError(
                    "The first element of a custom_input list element must be a string and the second element must be an image attachment or vice versa"
                )
        else:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list"
            )

    return collect_embedding_inputs(
        request,
        on_text=on_text,
        on_attachment=on_attachment,
        on_mixed=on_mixed,
    )


@dataclass
class MultiModalEmbeddingsAdapter(EmbeddingsAdapter):
    model_id: str
    client: GenAIClient
    storage: FileStorage | None

    @classmethod
    async def create(
        cls, storage: FileStorage | None, model_id: str, config: UpstreamConfig
    ) -> "EmbeddingsAdapter":
        client = await config.get_genai_client()
        return cls(model_id=model_id, client=client, storage=storage)

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        validate_request(request)

        base64_encode = request.encoding_format == "base64"

        sub_requests = [
            sub_request
            async for sub_request in await get_requests(self.storage, request)
        ]

        # NOTE: The model doesn't support batched inputs
        tasks = [
            compute_embeddings(
                sub_request,
                self.client,
                self.model_id,
                base64_encode=base64_encode,
                dimensions=request.dimensions,
            )
            for sub_request in sub_requests
        ]

        embeddings: list[Embedding] = []
        total_tokens = 0

        for embedding, tokens in await asyncio.gather(*tasks):
            embeddings.append(embedding)
            total_tokens += tokens

        return make_embeddings_response(
            model=self.model_id,
            embeddings=embeddings,
            tokens=total_tokens,
        )
