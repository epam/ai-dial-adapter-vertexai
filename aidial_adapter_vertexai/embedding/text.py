from logging import DEBUG
from typing import Dict, List, Optional, Tuple

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingInput

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs_without_attachments,
)
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.types import (
    Embedding,
    make_embeddings_response,
    vector_to_embedding,
)
from aidial_adapter_vertexai.utils.concurrency import make_async
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.vertex_ai import (
    TextEmbeddingModel,
    get_text_embedding_model,
    init_vertex_ai,
)

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
    model: TextEmbeddingModel,
    base64_encode: bool,
    dimensions: int | None,
    inputs: List[str | TextEmbeddingInput],
) -> Tuple[List[Embedding], int]:

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(
            {"inputs": inputs, "output_dimensionality": dimensions}
        )
        log.debug(f"request: {msg}")

    response = await make_async(
        lambda _: model.get_embeddings(
            inputs, output_dimensionality=dimensions
        ),
        (),
    )

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(response, excluded_keys=["_prediction_response"])
        log.debug(f"response: {msg}")

    embeddings: List[Embedding] = []
    tokens = 0

    for embedding in response:
        embeddings.append(vector_to_embedding(base64_encode, embedding.values))

        if embedding.statistics:
            tokens += embedding.statistics.token_count

    return embeddings, tokens


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

    async def on_texts(texts: List[str]) -> str | TextEmbeddingInput:
        if len(texts) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(texts) == 1:
            return texts[0]
        elif len(texts) == 2:
            title, text = texts
            if task_type != "RETRIEVAL_DOCUMENT":
                raise ValidationError(
                    "The model does not support inputs with titles "
                    "unless the type is RETRIEVAL_DOCUMENT"
                )
            return TextEmbeddingInput(
                title=title, text=text, task_type=task_type
            )
        else:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list - one for title and one for text."
            )

    iterator = collect_embedding_inputs_without_attachments(
        request, on_texts=on_texts
    )

    return [input async for input in iterator]


class TextEmbeddingsAdapter(EmbeddingsAdapter):
    model_id: str
    model: TextEmbeddingModel

    @classmethod
    async def create(
        cls,
        model_id: str,
        project_id: str,
        location: str,
    ) -> "EmbeddingsAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_text_embedding_model(model_id)
        return cls(model_id=model_id, model=model)

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

        embeddings, tokens = await compute_embeddings(
            self.model, base64_encode, request.dimensions, inputs
        )

        return make_embeddings_response(
            model=self.model_id,
            embeddings=embeddings,
            tokens=tokens,
        )
