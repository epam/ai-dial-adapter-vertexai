from enum import Enum


class ChatCompletionDeployment(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CHAT_BISON_2 = "chat-bison@002"
    CHAT_BISON_2_32K = "chat-bison-32k@002"

    CODECHAT_BISON_1 = "codechat-bison@001"
    CODECHAT_BISON_2 = "codechat-bison@002"
    CODECHAT_BISON_2_32K = "codechat-bison-32k@002"

    GEMINI_PRO_1 = "gemini-pro"
    GEMINI_PRO_VISION_1 = "gemini-pro-vision"
    GEMINI_PRO_VISION_1_5 = "gemini-1.5-pro-preview-0409"

    IMAGEN_005 = "imagegeneration@005"

    def get_model_id(self) -> str:
        return self.value


class EmbeddingsDeployment(str, Enum):
    TEXT_EMBEDDING_GECKO_1 = "textembedding-gecko@001"

    def get_model_id(self) -> str:
        return self.value
