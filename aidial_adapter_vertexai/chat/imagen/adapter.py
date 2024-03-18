from typing import List, Optional, Tuple

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
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer
from aidial_adapter_vertexai.vertex_ai import (
    get_image_generation_model,
    init_vertex_ai,
)

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
    ) -> Tuple[ImagenPrompt, List[int]]:
        return prompt, []

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
        data: str = image._as_base64_string()

        attachment: Attachment = Attachment(title="Image", type=type, data=data)

        if self.file_storage is not None:
            with Timer("upload to file storage: {time}", log.debug):
                meta = await self.file_storage.upload_file_as_base64(data, type)

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
        project_id: str,
        location: str,
    ) -> "ImagenChatCompletionAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_image_generation_model(model_id)
        return cls(file_storage, model)
