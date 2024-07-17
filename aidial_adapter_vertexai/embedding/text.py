from typing import Dict, Iterable, List, Optional, assert_never

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingInput

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.response import make_embeddings_response
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.encoding import vector_to_base64
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


async def get_text_embeddings(
    model_id: str,
    model: TextEmbeddingModel,
    base64_encode: bool,
    dimensions: int | None,
    inputs: List[str | TextEmbeddingInput],
) -> EmbeddingsResponse:

    embeddings = model.get_embeddings(inputs, output_dimensionality=dimensions)

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
        raise ValidationError("Request parameter 'dimensions' is not supported")

    if request.custom_fields is not None:
        if (
            not spec.supports_instr
            and request.custom_fields.instruction is not None
        ):
            raise ValidationError(
                "Request parameter 'custom_fields.instruction' is not supported"
            )

        if not spec.supports_type and request.custom_fields.type is not None:
            raise ValidationError(
                "Request parameter 'custom_fields.type' is not supported"
            )


def get_text(input: str | Attachment) -> str:
    if isinstance(input, str):
        return input
    else:
        raise ValidationError("Attachments are not supported")


def get_embedding_inputs(
    request: EmbeddingsRequest, task_type: Optional[str]
) -> Iterable[str | TextEmbeddingInput]:

    def make_input(
        text: str, title: str | None = None
    ) -> str | TextEmbeddingInput:
        if task_type is None and title is None:
            return text
        if title is not None and task_type != "RETRIEVAL_DOCUMENT":
            raise ValidationError(
                "The model does not support inputs with titles "
                "unless the type is RETRIEVAL_DOCUMENT"
            )
        return TextEmbeddingInput(title=title, text=text, task_type=task_type)

    if isinstance(request.input, str):
        yield make_input(request.input)
    elif isinstance(request.input, list):
        for input in request.input:
            if isinstance(input, str):
                yield make_input(input)
            else:
                raise ValidationError(
                    "Tokens in the input are not supported, provide text instead. "
                    "When Langchain AzureOpenAIEmbeddings class is used, set 'check_embedding_ctx_length=False' to disable tokenization."
                )
    else:
        assert_never(request.input)

    if request.custom_input is None:
        return

    for input in request.custom_input:
        if isinstance(input, (str, Attachment)):
            yield make_input(get_text(input))
        elif isinstance(input, list):
            if len(input) == 0:
                pass
            elif len(input) == 1:
                yield make_input(get_text(input[0]))
            elif len(input) == 2:
                yield make_input(
                    title=get_text(input[0]),
                    text=get_text(input[1]),
                )
            else:
                raise ValidationError(
                    "No more than two elements are allowed in an element of custom_input list - one for title and one for text."
                )
        else:
            assert_never(input)


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

        inputs: List[str | TextEmbeddingInput] = list(
            get_embedding_inputs(request, task_type)
        )

        base64_encode = request.encoding_format == "base64"

        return await get_text_embeddings(
            self.model_id,
            self.model,
            base64_encode,
            request.dimensions,
            inputs,
        )
