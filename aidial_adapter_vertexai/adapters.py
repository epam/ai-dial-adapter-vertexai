from typing import assert_never

from aidial_adapter_vertexai.chat.bison.adapter import (
    BisonChatAdapter,
    BisonCodeChatAdapter,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.claude.adapter import (
    ClaudeChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.gemini.adapter import (
    GeminiChatCompletionAdapter,
    GeminiGenAIChatCompletionAdapter,
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
from aidial_adapter_vertexai.upstream_config import UpstreamConfig


async def get_chat_completion_model(
    api_key: str,
    upstream_config: UpstreamConfig,
    deployment: ChatCompletionDeployment,
) -> ChatCompletionAdapter:

    model_id = deployment.get_model_id()
    storage = create_file_storage(api_key)

    match deployment:
        case (
            ChatCompletionDeployment.CHAT_BISON_1
            | ChatCompletionDeployment.CHAT_BISON_2
            | ChatCompletionDeployment.CHAT_BISON_2_32K
        ):
            upstream_config.dynamic_region_not_supported()
            return await BisonChatAdapter.create(model_id)
        case (
            ChatCompletionDeployment.CODECHAT_BISON_1
            | ChatCompletionDeployment.CODECHAT_BISON_2
            | ChatCompletionDeployment.CODECHAT_BISON_2_32K
        ):
            upstream_config.dynamic_region_not_supported()
            return await BisonCodeChatAdapter.create(model_id)
        case (
            ChatCompletionDeployment.GEMINI_PRO_1
            | ChatCompletionDeployment.GEMINI_PRO_VISION_1
            | ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW
            | ChatCompletionDeployment.GEMINI_PRO_1_5_V1
            | ChatCompletionDeployment.GEMINI_PRO_1_5_V2
            | ChatCompletionDeployment.GEMINI_FLASH_1_5_V1
            | ChatCompletionDeployment.GEMINI_FLASH_1_5_V2
        ):
            upstream_config.dynamic_region_not_supported()
            return await GeminiChatCompletionAdapter.create(
                storage, model_id, deployment
            )
        case (
            ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_001
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_01_21
            | ChatCompletionDeployment.GEMINI_2_0_PRO_EXP_02_05
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05
        ):
            return await GeminiGenAIChatCompletionAdapter.create(
                storage, model_id, deployment, location=upstream_config.region
            )
        case (
            ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2
            | ChatCompletionDeployment.CLAUDE_3_5_HAIKU
            | ChatCompletionDeployment.CLAUDE_3_OPUS
            | ChatCompletionDeployment.CLAUDE_3_5_SONNET
            | ChatCompletionDeployment.CLAUDE_3_HAIKU
        ):
            return await ClaudeChatCompletionAdapter.create(
                storage, model_id, deployment, region=upstream_config.region
            )
        case ChatCompletionDeployment.IMAGEN_005:
            upstream_config.dynamic_region_not_supported()
            return await ImagenChatCompletionAdapter.create(storage, model_id)
        case _:
            assert_never(deployment)


async def get_embeddings_model(
    api_key: str, deployment: EmbeddingsDeployment
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
            return await TextEmbeddingsAdapter.create(model_id)
        case EmbeddingsDeployment.MULTI_MODAL_EMBEDDING_1:
            storage = create_file_storage(api_key)
            return await MultiModalEmbeddingsAdapter.create(storage, model_id)
        case _:
            assert_never(deployment)
