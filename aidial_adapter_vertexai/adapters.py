from typing import Mapping, assert_never

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
from aidial_adapter_vertexai.embedding.gecko import (
    GeckoTextGenericEmbeddingsAdapter,
)


async def get_chat_completion_model(
    headers: Mapping[str, str],
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
            | ChatCompletionDeployment.GEMINI_PRO_1_5
            | ChatCompletionDeployment.GEMINI_FLASH_1_5
        ):
            storage = create_file_storage(headers)
            return await GeminiChatCompletionAdapter.create(
                storage, model_id, deployment, project_id, location
            )
        case ChatCompletionDeployment.IMAGEN_005:
            storage = create_file_storage(headers)
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
