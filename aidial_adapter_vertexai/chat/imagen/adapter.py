from typing import List, Optional

from aidial_sdk.chat_completion import Attachment, Message
from PIL import Image as PIL_Image
from typing_extensions import override
from vertexai.preview.vision_models import (
    GeneratedImage,
    ImageGenerationModel,
    ImageGenerationResponse,
)

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.dial_api.request import (
    ModelParameters,
    collect_text_content,
)
from aidial_adapter_vertexai.dial_api.storage import (
    FileStorage,
    compute_hash_digest,
)
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer
from aidial_adapter_vertexai.vertex_ai import get_image_generation_model

ImagenPrompt = str


class ImagenChatCompletionAdapter(ChatCompletionAdapter[ImagenPrompt]):
    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model: ImageGenerationModel,
    ):
        self.file_storage = file_storage
        self.model = model

    @override
    async def parse_prompt(
        self, tools: ToolsConfig, messages: List[Message]
    ) -> ImagenPrompt:
        tools.not_supported()

        if len(messages) == 0:
            raise ValidationError("The list of messages must not be empty")

        content = messages[-1].content
        if content is None:
            raise ValidationError("The last message must have content")

        return collect_text_content(content)

    @override
    async def truncate_prompt(
        self, prompt: ImagenPrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[ImagenPrompt]:
        return TruncatedPrompt(discarded_messages=[], prompt=prompt)

    @staticmethod
    def get_image_type(image: PIL_Image.Image) -> str:
        match image.format:
            case "JPEG":
                return "image/jpeg"
            case "PNG":
                return "image/png"
            case _:
                raise ValueError(f"Unknown image format: {image.format}")

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: ImagenPrompt
    ) -> None:
        prompt_tokens = await self.count_prompt_tokens(prompt)

        with Timer("predict timing: {time}", log.debug):
            response: ImageGenerationResponse = self.model.generate_images(
                prompt, number_of_images=1, seed=None
            )

        if len(response.images) == 0:
            raise RuntimeError("Expected 1 image in response, but got none")

        image: GeneratedImage = response[0]

        type: str = self.get_image_type(image._pil_image)
        data: bytes = image._image_bytes
        base64_data: str = image._as_base64_string()

        attachment: Attachment = Attachment(
            title="Image", type=type, data=base64_data
        )

        if self.file_storage is not None:
            with Timer("upload to file storage: {time}", log.debug):
                filename = "images/" + compute_hash_digest(base64_data)
                meta = await self.file_storage.upload(
                    filename=filename, content_type=type, content=data
                )

            attachment.data = None
            attachment.url = meta["url"]

        await consumer.add_attachment(attachment)

        # Avoid generating empty content
        completion = " "
        await consumer.append_content(completion)

        completion_tokens = await self.count_completion_tokens(completion)
        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

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
    ) -> "ImagenChatCompletionAdapter":
        model = await get_image_generation_model(model_id)
        return cls(file_storage, model)
