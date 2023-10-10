from aidial_adapter_vertexai.llm.bison_chat import (
    BisonChatAdapter,
    BisonCodeChatAdapter,
)
from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.embeddings_adapter import EmbeddingsAdapter
from aidial_adapter_vertexai.llm.gecko_embeddings import (
    GeckoTextClassificationEmbeddingsAdapter,
    GeckoTextClusteringEmbeddingsAdapter,
    GeckoTextGenericEmbeddingsAdapter,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters


async def get_chat_completion_model(
    deployment: ChatCompletionDeployment,
    project_id: str,
    location: str,
    model_params: ModelParameters,
) -> ChatCompletionAdapter:
    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            model_id = deployment.get_model_id()
            return await BisonChatAdapter.create(
                model_id, project_id, location, model_params
            )
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            model_id = deployment.get_model_id()
            return await BisonCodeChatAdapter.create(
                model_id, project_id, location, model_params
            )


async def get_embeddings_model(
    deployment: EmbeddingsDeployment,
    project_id: str,
    location: str,
) -> EmbeddingsAdapter:
    model_id = deployment.get_model_id()
    match deployment:
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1:
            return await GeckoTextGenericEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1_CLASSIFICATION:
            return await GeckoTextClassificationEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1_CLUSTERING:
            return await GeckoTextClusteringEmbeddingsAdapter.create(
                model_id, project_id, location
            )
