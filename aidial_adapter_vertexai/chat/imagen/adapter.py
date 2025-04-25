from typing import List, Optional

from aidial_sdk.chat_completion import Attachment, Message
from google.genai.client import Client as GenAIClient
from google.genai.types import (
    GenerateImagesConfigDict,
    GenerateImagesResponse,
    Image,
)
from PIL import Image as PIL_Image
from typing_extensions import override

from aidial_adapter_vertexai.app_config import get_genai_client
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
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
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.timer import Timer

ImagenPrompt = str


class ImagenChatCompletionAdapter(ChatCompletionAdapter[ImagenPrompt]):
    file_storage: FileStorage | None
    client: GenAIClient
    model_id: str

    def __init__(
        self,
        file_storage: FileStorage | None,
        client: GenAIClient,
        model_id: str,
    ):
        self.file_storage = file_storage
        self.client = client
        self.model_id = model_id

    @override
    async def parse_prompt(
        self,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> ImagenPrompt:
        tools.not_supported()
        static_tools.not_supported()
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

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: ImagenPrompt
    ) -> None:
        config = _prepare_generation_config(params)

        with Timer("predict timing: {time}", log.debug):
            response = await self.client.aio.models.generate_images(
                model=self.model_id, prompt=prompt, config=config
            )

        if (resource := _extract_image(response)) is None:
            raise RuntimeError("Expected image in response, but got none")

        attachment = Attachment(
            title="Image",
            type=resource.type,
            data=resource.data_base64,
        )

        if self.file_storage is not None:
            with Timer("upload to file storage: {time}", log.debug):
                filename = "images/" + compute_hash_digest(resource.data)
                meta = await self.file_storage.upload(
                    filename=filename,
                    content_type=resource.type,
                    content=resource.data,
                )

            attachment.data = None
            attachment.url = meta["url"]

        await consumer.add_attachment(attachment)

        # Avoid generating empty content
        completion = " "
        await consumer.append_content(completion)

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=await self.count_prompt_tokens(prompt),
                completion_tokens=await self.count_completion_tokens(
                    completion
                ),
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
        config: UpstreamConfig,
    ) -> "ImagenChatCompletionAdapter":
        return cls(
            file_storage,
            get_genai_client(config.project, config.region),
            model_id,
        )


def _prepare_generation_config(
    params: ModelParameters,
) -> GenerateImagesConfigDict:
    return {"seed": params.seed}


def _extract_image(
    response: GenerateImagesResponse,
) -> Resource | None:
    images = response.generated_images
    if images is None or len(images) == 0:
        return None

    image: Image | None = images[0].image
    if image is None:
        return None

    image_data: bytes | None = image.image_bytes
    if image_data is None:
        return None

    media_type: str = _get_image_type(image._pil_image)
    return Resource(type=media_type, data=image_data)


def _get_image_type(image: PIL_Image.Image) -> str:
    match image.format:
        case "JPEG":
            return "image/jpeg"
        case "PNG":
            return "image/png"
        case _:
            raise ValueError(f"Unknown image format: {image.format}")
