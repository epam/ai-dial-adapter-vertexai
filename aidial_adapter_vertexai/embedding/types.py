from typing import List

from aidial_sdk.embeddings import Embedding as SDKEmbedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage

from aidial_adapter_vertexai.embedding.encoding import vector_to_base64

Embedding = List[float] | str


def vector_to_embedding(base64_encode: bool, vector: List[float]) -> Embedding:
    return vector_to_base64(vector) if base64_encode else vector


def make_embeddings_response(
    model: str, vectors: List[Embedding], tokens: int
) -> EmbeddingsResponse:

    data: List[SDKEmbedding] = [
        SDKEmbedding(index=index, embedding=embedding)
        for index, embedding in enumerate(vectors)
    ]

    usage = Usage(
        prompt_tokens=tokens,
        total_tokens=tokens,
    )

    return EmbeddingsResponse(model=model, data=data, usage=usage)
