from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_vertexai.adapters import get_embeddings_model
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator


class VertexAIEmbeddings(Embeddings):
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region

    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:

        model = await get_embeddings_model(
            deployment=EmbeddingsDeployment(request.deployment_id),
            project_id=self.project_id,
            location=self.region,
            api_key=request.api_key,
        )

        return await model.embeddings(request)
