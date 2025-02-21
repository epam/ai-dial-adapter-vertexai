from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_vertexai.adapter_deployments import (
    AdapterEmbeddingsDeployment,
)
from aidial_adapter_vertexai.adapters import get_embeddings_model
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator


class VertexAIEmbeddings(Embeddings):
    deployment: AdapterEmbeddingsDeployment

    def __init__(self, deployment: AdapterEmbeddingsDeployment) -> None:
        self.deployment = deployment

    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:
        model = await get_embeddings_model(
            api_key=request.api_key, deployment=self.deployment
        )
        return await model.embeddings(request)
