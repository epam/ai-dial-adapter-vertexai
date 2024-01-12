from typing import Mapping, assert_never

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
from aidial_adapter_vertexai.llm.gemini_chat_completion_adapter import (
    GeminiChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.imagen_chat_completion_adapter import (
    ImagenChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_vertexai.universal_api.storage import create_file_storage


async def get_chat_completion_model(
    headers: Mapping[str, str],
    deployment: ChatCompletionDeployment,
    project_id: str,
    location: str,
) -> ChatCompletionAdapter:
    model_id = deployment.get_model_id()

    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1 | ChatCompletionDeployment.CHAT_BISON_2 | ChatCompletionDeployment.CHAT_BISON_2_32K:
            return await BisonChatAdapter.create(model_id, project_id, location)
        case ChatCompletionDeployment.CODECHAT_BISON_1 | ChatCompletionDeployment.CODECHAT_BISON_2 | ChatCompletionDeployment.CODECHAT_BISON_2_32K:
            return await BisonCodeChatAdapter.create(
                model_id, project_id, location
            )
        case ChatCompletionDeployment.GEMINI_PRO_1:
            return await GeminiChatCompletionAdapter.create(
                None, model_id, False, project_id, location
            )
        case ChatCompletionDeployment.GEMINI_PRO_VISION_1:
            storage = create_file_storage("images", headers)
            return await GeminiChatCompletionAdapter.create(
                storage, model_id, True, project_id, location
            )
        case ChatCompletionDeployment.IMAGEN_005:
            storage = create_file_storage("images", headers)
            return await ImagenChatCompletionAdapter.create(
                storage, model_id, project_id, location
            )
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
