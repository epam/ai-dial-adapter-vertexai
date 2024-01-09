from typing import AsyncIterator, List, Optional, Tuple

from aidial_sdk.chat_completion import Message
from typing_extensions import override
from vertexai.preview.generative_models import GenerativeModel

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.llm.vertex_ai import (
    get_gemini_model,
    init_vertex_ai,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.storage import FileStorage

ImagenPrompt = str


class ImagenChatCompletionAdapter(ChatCompletionAdapter[ImagenPrompt]):
    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model: GenerativeModel,
        is_vision_model: bool,
    ):
        self.file_storage = file_storage
        self.model = model
        self.is_vision_model = is_vision_model

    @override
    async def parse_prompt(self, messages: List[Message]) -> ImagenPrompt:
        if len(messages) == 0:
            raise ValidationError("The list of messages must not be empty")

        prompt = messages[-1].content
        if prompt is None:
            raise ValidationError("The last message must have content")

        return prompt

    @override
    async def truncate_prompt(
        self, prompt: ImagenPrompt, max_prompt_tokens: int
    ) -> Tuple[ImagenPrompt, int]:
        return prompt, 0

    async def send_message_async(
        self, params: ModelParameters, prompt: ImagenPrompt
    ) -> AsyncIterator[str]:
        yield "oops"

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: ImagenPrompt
    ) -> None:
        pass

    @override
    async def count_prompt_tokens(self, prompt: ImagenPrompt) -> int:
        return 0

    @override
    async def count_completion_tokens(self, string: str) -> int:
        return 1

    @classmethod
    async def create(
        cls,
        file_storage: Optional[FileStorage],
        model_id: str,
        has_vision: bool,
        project_id: str,
        location: str,
    ) -> "ImagenChatCompletionAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_gemini_model(model_id)
        return cls(file_storage, model, has_vision)
