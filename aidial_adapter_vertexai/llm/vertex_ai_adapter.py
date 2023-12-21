from typing import assert_never

from aidial_adapter_vertexai.llm.bison_adapter import (
    BisonChatAdapter,
    BisonCodeChatAdapter,
)
from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.embeddings_adapter import EmbeddingsAdapter
from aidial_adapter_vertexai.llm.gecko_embeddings import (
    GeckoTextGenericEmbeddingsAdapter,
)
from aidial_adapter_vertexai.llm.gemini_pro.adapter import GeminiProAdapter
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


async def get_chat_completion_model(
    deployment: ChatCompletionDeployment,
    project_id: str,
    location: str,
) -> ChatCompletionAdapter:
    model_id = deployment.get_model_id()
    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            return BisonChatAdapter.create(model_id, project_id, location)
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            return BisonCodeChatAdapter.create(model_id, project_id, location)
        case ChatCompletionDeployment.GEMINI_PRO_1:
            return GeminiProAdapter.create(model_id, project_id, location)
        case _:
            assert_never(deployment)


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
        case _:
            assert_never(deployment)
