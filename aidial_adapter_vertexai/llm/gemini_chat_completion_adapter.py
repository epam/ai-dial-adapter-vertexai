from typing import List, Tuple

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel
from typing_extensions import override
from vertexai.preview.generative_models import GenerativeModel

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.vertex_ai import (
    get_gemini_model,
    init_vertex_ai,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters


class GeminiPrompt(BaseModel):
    pass


class GeminiChatCompletionAdapter(ChatCompletionAdapter[GeminiPrompt]):
    def __init__(self, model: GenerativeModel):
        self.model = model

    @override
    async def parse_prompt(self, messages: List[Message]) -> GeminiPrompt:
        raise NotImplementedError()

    @override
    async def truncate_prompt(
        self, prompt: GeminiPrompt, max_prompt_tokens: int
    ) -> Tuple[GeminiPrompt, int]:
        raise NotImplementedError(
            "Prompt truncation is not supported for Genimi model yet"
        )

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: GeminiPrompt
    ) -> None:
        raise NotImplementedError()

    @override
    async def count_prompt_tokens(self, prompt: GeminiPrompt) -> int:
        raise NotImplementedError()

    @override
    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError()

    @override
    @classmethod
    async def create(
        cls, model_id: str, project_id: str, location: str
    ) -> "GeminiChatCompletionAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_gemini_model(model_id)
        return cls(model)
