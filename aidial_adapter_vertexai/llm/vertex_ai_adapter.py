from typing import assert_never

from vertexai.preview.language_models import ChatModel, CodeChatModel

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
from aidial_adapter_vertexai.llm.vertex_ai import init_vertex_ai
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


async def get_chat_completion_model(
    deployment: ChatCompletionDeployment, project_id: str, location: str
) -> ChatCompletionAdapter:
    model_id = deployment.get_model_id()

    async def get_chat():
        await init_vertex_ai(project_id, location)
        lang_model = ChatModel.from_pretrained(deployment)
        return BisonChatAdapter.create(
            lang_model, model_id, project_id, location
        )

    async def get_codechat():
        await init_vertex_ai(project_id, location)
        lang_model = CodeChatModel.from_pretrained(deployment)
        return BisonCodeChatAdapter.create(
            lang_model, model_id, project_id, location
        )

    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            return await get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2:
            return await get_chat()
        case ChatCompletionDeployment.CHAT_BISON_2_32K:
            return await get_chat()
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            return await get_codechat()
        case ChatCompletionDeployment.CODECHAT_BISON_2:
            return await get_codechat()
        case ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return await get_codechat()
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
