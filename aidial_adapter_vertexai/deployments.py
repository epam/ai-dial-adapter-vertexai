from enum import Enum
from typing import Literal


class ChatCompletionDeployment(Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CHAT_BISON_2 = "chat-bison@002"
    CHAT_BISON_2_32K = "chat-bison-32k@002"

    CODECHAT_BISON_1 = "codechat-bison@001"
    CODECHAT_BISON_2 = "codechat-bison@002"
    CODECHAT_BISON_2_32K = "codechat-bison-32k@002"

    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_1 = "gemini-1.0-pro"
    GEMINI_PRO_VISION_1 = "gemini-pro-vision"
    GEMINI_PRO_1_5_PREVIEW = "gemini-1.5-pro-preview-0409"
    GEMINI_PRO_1_5_V1 = "gemini-1.5-pro-001"
    GEMINI_PRO_1_5_V2 = "gemini-1.5-pro-002"
    GEMINI_FLASH_1_5_V1 = "gemini-1.5-flash-001"
    GEMINI_FLASH_1_5_V2 = "gemini-1.5-flash-002"
    GEMINI_2_0_FLASH_THINKING_EXP_01_21 = "gemini-2.0-flash-thinking-exp-01-21"
    GEMINI_2_0_PRO_EXP_02_05 = "gemini-2.0-pro-exp-02-05"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_001 = "gemini-2.0-flash-001"

    GEMINI_2_0_FLASH_LITE_PREVIEW_02_05 = "gemini-2.0-flash-lite-preview-02-05"

    IMAGEN_005 = "imagegeneration@005"

    CLAUDE_3_5_SONNET_V2 = "claude-3-5-sonnet-v2@20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku@20241022"
    CLAUDE_3_OPUS = "claude-3-opus@20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet@20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku@20240307"


# Redirect deprecated 'gemini-pro' alias to 'gemini-1.0-pro'
CHAT_COMPLETION_REDIRECTS = {
    ChatCompletionDeployment.GEMINI_PRO: ChatCompletionDeployment.GEMINI_PRO_1
}


ClaudeDeployment = Literal[
    ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2,
    ChatCompletionDeployment.CLAUDE_3_5_HAIKU,
    ChatCompletionDeployment.CLAUDE_3_OPUS,
    ChatCompletionDeployment.CLAUDE_3_5_SONNET,
    ChatCompletionDeployment.CLAUDE_3_HAIKU,
]

GeminiDeployment = Literal[
    ChatCompletionDeployment.GEMINI_PRO,
    ChatCompletionDeployment.GEMINI_PRO_1,
    ChatCompletionDeployment.GEMINI_PRO_VISION_1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V1,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
]


Gemini2Deployment = Literal[
    ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_001,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
    ChatCompletionDeployment.GEMINI_2_0_PRO_EXP_02_05,
    ChatCompletionDeployment.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
]


class EmbeddingsDeployment(Enum):
    # English models
    TEXT_EMBEDDING_GECKO_1 = "textembedding-gecko@001"
    TEXT_EMBEDDING_GECKO_3 = "textembedding-gecko@003"
    TEXT_EMBEDDING_4 = "text-embedding-004"

    # Multilingual models
    TEXT_EMBEDDING_GECKO_MULTILINGUAL_1 = "textembedding-gecko-multilingual@001"
    TEXT_MULTILINGUAL_EMBEDDING_2 = "text-multilingual-embedding-002"

    MULTI_MODAL_EMBEDDING_1 = "multimodalembedding@001"


TextEmbeddingDeployment = Literal[
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1,
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_3,
    EmbeddingsDeployment.TEXT_EMBEDDING_4,
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1,
    EmbeddingsDeployment.TEXT_MULTILINGUAL_EMBEDDING_2,
]
