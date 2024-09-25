from typing import assert_never

from aidial_adapter_vertexai.chat.bison.adapter import (
    BisonChatAdapter,
    BisonCodeChatAdapter,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.gemini.adapter import (
    GeminiChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.imagen.adapter import (
    ImagenChatCompletionAdapter,
)
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_vertexai.dial_api.storage import create_file_storage
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.multi_modal import (
    MultiModalEmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.text import TextEmbeddingsAdapter


async def get_chat_completion_model(
    api_key: str,
    deployment: ChatCompletionDeployment,
    project_id: str,
    location: str,
) -> ChatCompletionAdapter:
    model_id = deployment.get_model_id()

    match deployment:
        case (
            ChatCompletionDeployment.CHAT_BISON_1
            | ChatCompletionDeployment.CHAT_BISON_2
            | ChatCompletionDeployment.CHAT_BISON_2_32K
        ):
            return await BisonChatAdapter.create(model_id, project_id, location)
        case (
            ChatCompletionDeployment.CODECHAT_BISON_1
            | ChatCompletionDeployment.CODECHAT_BISON_2
            | ChatCompletionDeployment.CODECHAT_BISON_2_32K
        ):
            return await BisonCodeChatAdapter.create(
                model_id, project_id, location
            )
        case (
            ChatCompletionDeployment.GEMINI_PRO_1
            | ChatCompletionDeployment.GEMINI_PRO_VISION_1
            | ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW
            | ChatCompletionDeployment.GEMINI_PRO_1_5_V1
            | ChatCompletionDeployment.GEMINI_PRO_1_5_V2
            | ChatCompletionDeployment.GEMINI_FLASH_1_5_V1
            | ChatCompletionDeployment.GEMINI_FLASH_1_5_V2
        ):
            storage = create_file_storage(api_key)
            return await GeminiChatCompletionAdapter.create(
                storage, model_id, deployment, project_id, location
            )
        case ChatCompletionDeployment.IMAGEN_005:
            storage = create_file_storage(api_key)
            return await ImagenChatCompletionAdapter.create(
                storage, model_id, project_id, location
            )
        case _:
            assert_never(deployment)


async def get_embeddings_model(
    deployment: EmbeddingsDeployment,
    project_id: str,
    location: str,
    api_key: str,
) -> EmbeddingsAdapter:
    model_id = deployment.get_model_id()
    match deployment:
        case (
            EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1
            | EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_3
            | EmbeddingsDeployment.TEXT_EMBEDDING_4
            | EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1
            | EmbeddingsDeployment.TEXT_MULTILINGUAL_EMBEDDING_2
        ):
            return await TextEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case EmbeddingsDeployment.MULTI_MODAL_EMBEDDING_1:
            return await MultiModalEmbeddingsAdapter.create(
                model_id, project_id, location, api_key
            )
        case _:
            assert_never(deployment)
