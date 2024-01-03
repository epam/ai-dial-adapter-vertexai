from enum import Enum


class ChatCompletionDeployment(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CODECHAT_BISON_1 = "codechat-bison@001"
    GEMINI_PRO_1 = "gemini-pro"

    def get_model_id(self) -> str:
        return self.value


class EmbeddingsDeployment(str, Enum):
    TEXT_EMBEDDING_GECKO_1 = "textembedding-gecko@001"

    def get_model_id(self) -> str:
        return self.value
