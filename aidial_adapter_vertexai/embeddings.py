from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_vertexai.adapters import get_embeddings_model
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.upstream_config import parse_upstream_config
from aidial_adapter_vertexai.utils.adapter_deployments import (
    resolve_upstream_deployment_id_from_request,
)


class VertexAIEmbeddings(Embeddings):
    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:
        deployment = resolve_upstream_deployment_id_from_request(
            EmbeddingsDeployment, request
        )
        model = await get_embeddings_model(
            api_key=request.api_key,
            deployment=deployment,
            upstream_config=parse_upstream_config(request.original_request),
        )
        return await model.embeddings(request)
