from logging import DEBUG
from typing import Dict, List, Optional

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingInput

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    collect_embedding_inputs_no_attachments,
)
from aidial_adapter_vertexai.dial_api.response import make_embeddings_response
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.encoding import vector_to_base64
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.vertex_ai import TextEmbeddingModel

# See available task types at: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype
# The list of task types tends to grow with time,
# so we don't try to validate it here.


class ModelSpec(BaseModel):
    supports_type: bool
    supports_instr: bool
    supports_dimensions: bool


specs: Dict[str, ModelSpec] = {
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1: ModelSpec(
        supports_type=False,
        supports_instr=False,
        supports_dimensions=False,
    ),
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_3: ModelSpec(
        supports_type=True,
        supports_instr=False,
        supports_dimensions=False,
    ),
    EmbeddingsDeployment.TEXT_EMBEDDING_4: ModelSpec(
        supports_type=True,
        supports_instr=False,
        supports_dimensions=True,
    ),
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1: ModelSpec(
        supports_type=True,
        supports_instr=False,
        supports_dimensions=False,
    ),
    EmbeddingsDeployment.TEXT_MULTILINGUAL_EMBEDDING_2: ModelSpec(
        supports_type=True,
        supports_instr=False,
        supports_dimensions=True,
    ),
}


async def compute_embeddings(
    model_id: str,
    model: TextEmbeddingModel,
    base64_encode: bool,
    dimensions: int | None,
    inputs: List[str | TextEmbeddingInput],
) -> EmbeddingsResponse:

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(
            {"inputs": inputs, "output_dimensionality": dimensions}
        )
        log.debug(f"request: {msg}")

    embeddings = model.get_embeddings(inputs, output_dimensionality=dimensions)

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(
            embeddings, excluded_keys=["_prediction_response"]
        )
        log.debug(f"response: {msg}")

    vectors: List[List[float] | str] = []
    token_count = 0

    for embedding in embeddings:
        vectors.append(
            vector_to_base64(embedding.values)
            if base64_encode
            else embedding.values
        )

        if embedding.statistics:
            token_count += embedding.statistics.token_count

    return make_embeddings_response(
        model=model_id,
        vectors=vectors,
        usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
    )


def validate_request(spec: ModelSpec, request: EmbeddingsRequest) -> None:
    if not spec.supports_dimensions and request.dimensions:
        raise ValidationError("Dimensions parameter is not supported")

    if request.custom_fields is not None:
        if (
            not spec.supports_instr
            and request.custom_fields.instruction is not None
        ):
            raise ValidationError("Instruction prompt is not supported")

        if not spec.supports_type and request.custom_fields.type is not None:
            raise ValidationError(
                "The embedding model does not support embedding types"
            )


async def get_embedding_inputs(
    request: EmbeddingsRequest, task_type: Optional[str]
) -> List[str | TextEmbeddingInput]:

    async def on_text(text: str) -> str | TextEmbeddingInput:
        return text

    async def on_texts(
        fst: str, snd: str, rest: List[str]
    ) -> str | TextEmbeddingInput:
        if rest != []:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list - one for title and one for text."
            )

        if task_type != "RETRIEVAL_DOCUMENT":
            raise ValidationError(
                "The model does not support inputs with titles "
                "unless the type is RETRIEVAL_DOCUMENT"
            )
        return TextEmbeddingInput(title=fst, text=snd, task_type=task_type)

    iterator = collect_embedding_inputs_no_attachments(
        request, on_text=on_text, on_texts=on_texts
    )

    return [input async for input in iterator]


class TextEmbeddingsAdapter(EmbeddingsAdapter):
    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        spec = specs.get(self.model_id)
        if spec is None:
            raise RuntimeError(
                f"Can't find the model {self.model_id!r} in the specs"
            )

        validate_request(spec, request)

        task_type: Optional[str] = None
        if request.custom_fields is not None:
            task_type = request.custom_fields.type

        inputs = await get_embedding_inputs(request, task_type)

        base64_encode = request.encoding_format == "base64"

        return await compute_embeddings(
            self.model_id,
            self.model,
            base64_encode,
            request.dimensions,
            inputs,
        )
