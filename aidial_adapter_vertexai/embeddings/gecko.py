from enum import Enum
from typing import List, Optional, Tuple

from vertexai.language_models import TextEmbeddingInput

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.request import EmbeddingsType
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.embeddings.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.vertex_ai import TextEmbeddingModel


class GeckoEmbeddingType(str, Enum):
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"


def to_gecko_embedding_type(
    ty: EmbeddingsType,
) -> Optional[GeckoEmbeddingType]:
    match ty:
        case EmbeddingsType.SYMMETRIC:
            return None
        case EmbeddingsType.DOCUMENT:
            return GeckoEmbeddingType.RETRIEVAL_DOCUMENT
        case EmbeddingsType.QUERY:
            return GeckoEmbeddingType.RETRIEVAL_QUERY


async def get_gecko_embeddings(
    model: TextEmbeddingModel,
    input: str | List[str],
    task_type: Optional[GeckoEmbeddingType],
) -> Tuple[List[List[float]], TokenUsage]:
    inputs = [input] if isinstance(input, str) else input
    texts: List[str | TextEmbeddingInput] = [
        TextEmbeddingInput(text=text, task_type=task_type) for text in inputs
    ]

    embeddings = model.get_embeddings(texts)

    vectors = [embedding.values for embedding in embeddings]
    token_count = sum(
        embedding.statistics.token_count for embedding in embeddings  # type: ignore
    )

    return vectors, TokenUsage(prompt_tokens=token_count, completion_tokens=0)


def validate_parameters(
    embedding_type: EmbeddingsType,
    embedding_instruction: Optional[str],
    supported_embedding_types: List[EmbeddingsType],
) -> None:
    if embedding_instruction is not None:
        raise ValidationError("Instruction prompt is not supported")

    assert (
        len(supported_embedding_types) != 0
    ), "The embedding model doesn't support any embedding types"

    if embedding_type not in supported_embedding_types:
        allowed = ", ".join([e.value for e in supported_embedding_types])
        raise ValidationError(
            f"Embedding types other than {allowed} are not supported"
        )


class GeckoTextGenericEmbeddingsAdapter(EmbeddingsAdapter):
    async def embeddings(
        self,
        input: str | List[str],
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        validate_parameters(
            embedding_type, embedding_instruction, [EmbeddingsType.SYMMETRIC]
        )

        task_type = to_gecko_embedding_type(embedding_type)
        return await get_gecko_embeddings(self.model, input, task_type)
