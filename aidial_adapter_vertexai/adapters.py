from typing import assert_never

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.claude.adapter import (
    ClaudeChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.gemini.adapter import (
    GeminiGenAIChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.imagen.adapter import (
    ImagenChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.mistral.adapter import (
    MistralChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.veo.adapter import VeoChatCompletionAdapter
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.deployments import EmbeddingsDeployment as E
from aidial_adapter_vertexai.dial_api.storage import create_file_storage
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.genai import GenAIEmbeddingsAdapter
from aidial_adapter_vertexai.embedding.multi_modal import (
    MultiModalEmbeddingsAdapter,
)
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.adapter_deployments import (
    AdapterChatCompletionDeployment,
    AdapterEmbeddingsDeployment,
)


async def get_chat_completion_model(
    *,
    api_key: str,
    upstream_config: UpstreamConfig,
    deployment: AdapterChatCompletionDeployment,
) -> ChatCompletionAdapter:
    storage = create_file_storage(api_key)

    match deployment.reference_deployment_id:
        case (
            D.GEMINI_2_0_FLASH_EXP
            | D.GEMINI_2_0_FLASH_001
            | D.GEMINI_2_5_PRO
            | D.GEMINI_2_5_PRO_PREVIEW_03_25
            | D.GEMINI_2_0_FLASH_LITE_1
            | D.GEMINI_2_5_FLASH
            | D.GEMINI_2_5_FLASH_IMAGE_PREVIEW
            | D.GEMINI_2_5_FLASH_IMAGE
            | D.GEMINI_3_PRO
            | D.GEMINI_3_PRO_PREVIEW
            | D.GEMINI_3_FLASH_PREVIEW
            | D.GEMINI_3_PRO_IMAGE_PREVIEW
            | D.GEMINI_3_1_PRO_PREVIEW
            | D.GEMINI_3_1_FLASH_IMAGE_PREVIEW
            | D.GEMINI_3_1_FLASH_LITE_PREVIEW
            | D.GEMINI_3_1_FLASH_LITE
            | D.GEMINI_3_5_FLASH
        ):
            return await GeminiGenAIChatCompletionAdapter.create(
                storage,
                deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case (
            D.CLAUDE_3_5_SONNET_V2
            | D.CLAUDE_3_5_HAIKU
            | D.CLAUDE_3_OPUS
            | D.CLAUDE_3_5_SONNET
            | D.CLAUDE_3_HAIKU
            | D.CLAUDE_3_7_SONNET
            | D.CLAUDE_4_OPUS
            | D.CLAUDE_4_SONNET
            | D.CLAUDE_4_1_OPUS
            | D.CLAUDE_4_5_HAIKU
            | D.CLAUDE_4_5_SONNET
            | D.CLAUDE_4_6_SONNET
            | D.CLAUDE_4_6_OPUS
            | D.CLAUDE_4_5_OPUS
        ):
            return await ClaudeChatCompletionAdapter.create(
                storage,
                deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case (
            D.IMAGEN_005
            | D.IMAGEN_3_GENERATE_001
            | D.IMAGEN_3_GENERATE_002
            | D.IMAGEN_3_FAST_GENERATE
            | D.IMAGEN_4_GENERATE_PREVIEW
            | D.IMAGEN_4_FAST_GENERATE_PREVIEW
            | D.IMAGEN_4_ULTRA_GENERATE_PREVIEW
            | D.IMAGEN_4_GENERATE
            | D.IMAGEN_4_FAST_GENERATE
            | D.IMAGEN_4_ULTRA_GENERATE
        ):
            return await ImagenChatCompletionAdapter.create(
                storage,
                deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case (
            D.VEO_3_0_GENERATE
            | D.VEO_3_0_GENERATE_PREVIEW
            | D.VEO_3_0_FAST_GENERATE
            | D.VEO_3_0_FAST_GENERATE_PREVIEW
            | D.VEO_3_1_GENERATE
            | D.VEO_3_1_GENERATE_PREVIEW
            | D.VEO_3_1_FAST_GENERATE
            | D.VEO_3_1_FAST_GENERATE_PREVIEW
        ):
            return await VeoChatCompletionAdapter.create(
                storage,
                deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case D.MISTRAL_MEDIUM_3 | D.MISTRAL_SMALL | D.MISTRAL_CODESTRAL_2:
            return await MistralChatCompletionAdapter.create(
                storage,
                deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case _:
            assert_never(deployment.reference_deployment_id)


async def get_embeddings_model(
    *,
    api_key: str,
    upstream_config: UpstreamConfig,
    deployment: AdapterEmbeddingsDeployment,
) -> EmbeddingsAdapter:
    storage = create_file_storage(api_key)

    match deployment.reference_deployment_id:
        case (
            E.TEXT_EMBEDDING_GECKO_1
            | E.TEXT_EMBEDDING_GECKO_3
            | E.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1
            | E.TEXT_GEMINI_EMBEDDING_1
            | E.TEXT_EMBEDDING_4
            | E.TEXT_EMBEDDING_5
            | E.TEXT_MULTILINGUAL_EMBEDDING_2
            | E.GEMINI_EMBEDDING_2_PREVIEW
        ):
            return await GenAIEmbeddingsAdapter.create(
                storage=storage,
                deployment=deployment.clone(deployment.reference_deployment_id),
                config=upstream_config,
            )
        case E.MULTI_MODAL_EMBEDDING_1:
            return await MultiModalEmbeddingsAdapter.create(
                storage, deployment.upstream_deployment_id
            )
        case _:
            assert_never(deployment)
