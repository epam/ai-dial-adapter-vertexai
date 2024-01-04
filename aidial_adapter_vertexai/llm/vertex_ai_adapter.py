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
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


async def get_chat_completion_model(
    deployment: ChatCompletionDeployment, project_id: str, location: str
) -> ChatCompletionAdapter:
    model_id = deployment.get_model_id()

    def get_chat():
        return BisonChatAdapter.create(model_id, project_id, location)

    def get_codechat():
        return BisonCodeChatAdapter.create(model_id, project_id, location)

    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            return get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2:
            return get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2_32K:
            return get_chat()
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            return BisonCodeChatAdapter.create(model_id, project_id, location)
        case ChatCompletionDeployment.CODECHAT_BISON_2:
            return get_codechat()
        case ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return get_codechat()
        case ChatCompletionDeployment.GEMINI_PRO_1:
            raise NotImplementedError("Gemini Pro is not supported yet")
        case _:
            assert_never(deployment)


async def get_embeddings_model(
    deployment: EmbeddingsDeployment, project_id: str, location: str
) -> EmbeddingsAdapter:
    model_id = deployment.get_model_id()
    match deployment:
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1:
            return await GeckoTextGenericEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case _:
            assert_never(deployment)
