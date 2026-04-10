from aidial_sdk.embeddings import Embedding as SDKEmbedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage

from aidial_adapter_vertexai.embedding.encoding import vector_to_base64

Embedding = list[float] | str


def vector_to_embedding(base64_encode: bool, vector: list[float]) -> Embedding:
    return vector_to_base64(vector) if base64_encode else vector


def make_embeddings_response(
    model: str, embeddings: list[Embedding], tokens: int
) -> EmbeddingsResponse:
    data: list[SDKEmbedding] = [
        SDKEmbedding(index=index, embedding=embedding)
        for index, embedding in enumerate(embeddings)
    ]

    usage = Usage(
        prompt_tokens=tokens,
        total_tokens=tokens,
    )

    return EmbeddingsResponse(model=model, data=data, usage=usage)
