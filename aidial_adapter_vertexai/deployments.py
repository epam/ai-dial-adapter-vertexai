from enum import Enum
from typing import Literal


class ChatCompletionDeployment(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CHAT_BISON_2 = "chat-bison@002"
    CHAT_BISON_2_32K = "chat-bison-32k@002"

    CODECHAT_BISON_1 = "codechat-bison@001"
    CODECHAT_BISON_2 = "codechat-bison@002"
    CODECHAT_BISON_2_32K = "codechat-bison-32k@002"

    GEMINI_PRO_1 = "gemini-pro"
    GEMINI_PRO_VISION_1 = "gemini-pro-vision"
    GEMINI_PRO_1_5_PREVIEW = "gemini-1.5-pro-preview-0409"
    GEMINI_PRO_1_5_V1 = "gemini-1.5-pro-001"
    GEMINI_PRO_1_5_V2 = "gemini-1.5-pro-002"
    GEMINI_FLASH_1_5_V1 = "gemini-1.5-flash-001"
    GEMINI_FLASH_1_5_V2 = "gemini-1.5-flash-002"

    IMAGEN_005 = "imagegeneration@005"

    def get_model_id(self) -> str:
        return self.value


GeminiDeployment = Literal[
    ChatCompletionDeployment.GEMINI_PRO_1,
    ChatCompletionDeployment.GEMINI_PRO_VISION_1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V1,
    ChatCompletionDeployment.GEMINI_PRO_1_5_V2,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V1,
    ChatCompletionDeployment.GEMINI_FLASH_1_5_V2,
]


class EmbeddingsDeployment(str, Enum):
    # English models
    TEXT_EMBEDDING_GECKO_1 = "textembedding-gecko@001"
    TEXT_EMBEDDING_GECKO_3 = "textembedding-gecko@003"
    TEXT_EMBEDDING_4 = "text-embedding-004"

    # Multilingual models
    TEXT_EMBEDDING_GECKO_MULTILINGUAL_1 = "textembedding-gecko-multilingual@001"
    TEXT_MULTILINGUAL_EMBEDDING_2 = "text-multilingual-embedding-002"

    MULTI_MODAL_EMBEDDING_1 = "multimodalembedding@001"

    def get_model_id(self) -> str:
        return self.value
