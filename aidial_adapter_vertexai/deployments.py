from enum import Enum
from typing import List, Literal, Self


class DeploymentEnum(Enum):
    @classmethod
    def deployments(cls) -> List[str]:
        return [d.value for d in cls]

    @classmethod
    def from_string(cls, model_id: str) -> Self | None:
        for deployment in cls:
            if deployment.value == model_id:
                return deployment
        return None


class ChatCompletionDeployment(DeploymentEnum):
    GEMINI_2_0_FLASH_LITE_1 = "gemini-2.0-flash-lite-001"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_001 = "gemini-2.0-flash-001"

    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_PRO_PREVIEW_03_25 = "gemini-2.5-pro-preview-03-25"

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_IMAGE_PREVIEW = "gemini-2.5-flash-image-preview"
    GEMINI_2_5_FLASH_IMAGE = "gemini-2.5-flash-image"
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_PRO_IMAGE_PREVIEW = "gemini-3-pro-image-preview"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"

    IMAGEN_005 = "imagegeneration@005"
    IMAGEN_3_GENERATE_001 = "imagen-3.0-generate-001"
    IMAGEN_3_GENERATE_002 = "imagen-3.0-generate-002"
    IMAGEN_3_FAST_GENERATE = "imagen-3.0-fast-generate-001"

    IMAGEN_4_GENERATE_PREVIEW = "imagen-4.0-generate-preview-06-06"
    IMAGEN_4_FAST_GENERATE_PREVIEW = "imagen-4.0-fast-generate-preview-06-06"
    IMAGEN_4_ULTRA_GENERATE_PREVIEW = "imagen-4.0-ultra-generate-preview-06-06"
    IMAGEN_4_GENERATE = "imagen-4.0-generate-001"
    IMAGEN_4_FAST_GENERATE = "imagen-4.0-fast-generate-001"
    IMAGEN_4_ULTRA_GENERATE = "imagen-4.0-ultra-generate-001"

    CLAUDE_3_5_SONNET_V2 = "claude-3-5-sonnet-v2@20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku@20241022"
    CLAUDE_3_OPUS = "claude-3-opus@20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet@20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku@20240307"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet@20250219"
    CLAUDE_4_OPUS = "claude-opus-4@20250514"
    CLAUDE_4_SONNET = "claude-sonnet-4@20250514"
    CLAUDE_4_1_OPUS = "claude-opus-4-1@20250805"
    CLAUDE_4_5_HAIKU = "claude-haiku-4-5@20251001"
    CLAUDE_4_5_SONNET = "claude-sonnet-4-5@20250929"

    VEO_3_0_GENERATE = "veo-3.0-generate-001"
    VEO_3_0_GENERATE_PREVIEW = "veo-3.0-generate-preview"
    VEO_3_0_FAST_GENERATE = "veo-3.0-fast-generate-001"
    VEO_3_0_FAST_GENERATE_PREVIEW = "veo-3.0-fast-generate-preview"
    VEO_3_1_GENERATE = "veo-3.1-generate-001"
    VEO_3_1_GENERATE_PREVIEW = "veo-3.1-generate-preview"
    VEO_3_1_FAST_GENERATE = "veo-3.1-fast-generate-001"
    VEO_3_1_FAST_GENERATE_PREVIEW = "veo-3.1-fast-generate-preview"


ImagenDeployment = Literal[
    ChatCompletionDeployment.IMAGEN_005,
    ChatCompletionDeployment.IMAGEN_3_GENERATE_001,
    ChatCompletionDeployment.IMAGEN_3_GENERATE_002,
    ChatCompletionDeployment.IMAGEN_3_FAST_GENERATE,
    ChatCompletionDeployment.IMAGEN_4_GENERATE_PREVIEW,
    ChatCompletionDeployment.IMAGEN_4_FAST_GENERATE_PREVIEW,
    ChatCompletionDeployment.IMAGEN_4_ULTRA_GENERATE_PREVIEW,
    ChatCompletionDeployment.IMAGEN_4_GENERATE,
    ChatCompletionDeployment.IMAGEN_4_FAST_GENERATE,
    ChatCompletionDeployment.IMAGEN_4_ULTRA_GENERATE,
]

ClaudeDeployment = Literal[
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_OPUS,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET,
    ChatCompletionDeployment.CLAUDE_3_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_7_SONNET,
    ChatCompletionDeployment.CLAUDE_4_OPUS,
    ChatCompletionDeployment.CLAUDE_4_SONNET,
    ChatCompletionDeployment.CLAUDE_4_1_OPUS,
    ChatCompletionDeployment.CLAUDE_4_5_HAIKU,
    ChatCompletionDeployment.CLAUDE_4_5_SONNET,
]

GeminiDeployment = Literal[
    ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_001,
    ChatCompletionDeployment.GEMINI_2_5_PRO,
    ChatCompletionDeployment.GEMINI_2_5_PRO_PREVIEW_03_25,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_LITE_1,
    ChatCompletionDeployment.GEMINI_2_5_FLASH,
    ChatCompletionDeployment.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
    ChatCompletionDeployment.GEMINI_2_5_FLASH_IMAGE,
    ChatCompletionDeployment.GEMINI_3_PRO,
    ChatCompletionDeployment.GEMINI_3_PRO_PREVIEW,
    ChatCompletionDeployment.GEMINI_3_FLASH_PREVIEW,
    ChatCompletionDeployment.GEMINI_3_PRO_IMAGE_PREVIEW,
    ChatCompletionDeployment.GEMINI_3_1_PRO_PREVIEW,
]

VeoDeployment = Literal[
    ChatCompletionDeployment.VEO_3_0_GENERATE,
    ChatCompletionDeployment.VEO_3_0_GENERATE_PREVIEW,
    ChatCompletionDeployment.VEO_3_0_FAST_GENERATE,
    ChatCompletionDeployment.VEO_3_0_FAST_GENERATE_PREVIEW,
    ChatCompletionDeployment.VEO_3_1_GENERATE,
    ChatCompletionDeployment.VEO_3_1_GENERATE_PREVIEW,
    ChatCompletionDeployment.VEO_3_1_FAST_GENERATE,
    ChatCompletionDeployment.VEO_3_1_FAST_GENERATE_PREVIEW,
]


class EmbeddingsDeployment(DeploymentEnum):
    # Text: English models
    TEXT_EMBEDDING_GECKO_1 = "textembedding-gecko@001"
    TEXT_EMBEDDING_GECKO_3 = "textembedding-gecko@003"
    TEXT_EMBEDDING_4 = "text-embedding-004"
    TEXT_EMBEDDING_5 = "text-embedding-005"

    # Text: Multilingual models
    TEXT_EMBEDDING_GECKO_MULTILINGUAL_1 = "textembedding-gecko-multilingual@001"
    TEXT_MULTILINGUAL_EMBEDDING_2 = "text-multilingual-embedding-002"
    TEXT_GEMINI_EMBEDDING_1 = "gemini-embedding-001"

    # Text/Image models
    MULTI_MODAL_EMBEDDING_1 = "multimodalembedding@001"


TextEmbeddingDeployment = Literal[
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1,
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_3,
    EmbeddingsDeployment.TEXT_EMBEDDING_4,
    EmbeddingsDeployment.TEXT_EMBEDDING_5,
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1,
    EmbeddingsDeployment.TEXT_MULTILINGUAL_EMBEDDING_2,
    EmbeddingsDeployment.TEXT_GEMINI_EMBEDDING_1,
]
