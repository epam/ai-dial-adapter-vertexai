from abc import ABC, abstractmethod

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_vertexai.vertex_ai import (
    TextEmbeddingModel,
    get_embedding_model,
    init_vertex_ai,
)


class EmbeddingsAdapter(ABC):
    def __init__(self, model_id: str, model: TextEmbeddingModel):
        self.model_id = model_id
        self.model = model

    @classmethod
    async def create(cls, model_id: str, project_id: str, location: str):
        await init_vertex_ai(project_id, location)
        model = await get_embedding_model(model_id)
        return cls(model_id, model)

    @abstractmethod
    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        pass
