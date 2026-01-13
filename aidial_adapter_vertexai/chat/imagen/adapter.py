from logging import DEBUG
from typing import List, Optional

from aidial_sdk.chat_completion import Attachment, Message
from aidial_sdk.exceptions import InvalidRequestError
from google.genai.client import Client as GenAIClient
from google.genai.types import GenerateImagesConfigDict, GenerateImagesResponse
from PIL import Image as PIL_Image
from pydantic import BaseModel
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.imagen.configuration import ImagenConfig
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import ImagenDeployment
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
from aidial_adapter_vertexai.utils.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.timer import Timer

ImagenPrompt = str


class ImagenChatCompletionAdapter(ChatCompletionAdapter[ImagenPrompt]):
    file_storage: FileStorage | None
    client: GenAIClient
    deployment: AdapterDeployment[ImagenDeployment]

    def __init__(
        self,
        file_storage: FileStorage | None,
        client: GenAIClient,
        deployment: AdapterDeployment[ImagenDeployment],
    ):
        self.file_storage = file_storage
        self.client = client
        self.deployment = deployment

    @property
    def model_id(self) -> str:
        return self.deployment.upstream_deployment_id

    async def configuration(self) -> type[ImagenConfig]:
        return ImagenConfig

    @override
    async def parse_prompt(
        self,
        params: ModelParameters,
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
        configuration = params.parse_configuration(await self.configuration())
        config = _prepare_generation_config(params, configuration)

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"model": self.model_id, "prompt": prompt, "config": config}
            )
            log.debug(f"request: {msg}")

        with Timer("predict timing: {time}", log.debug):
            response = await self.client.aio.models.generate_images(
                model=self.model_id, prompt=prompt, config=config
            )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(response)}")

        if (generated_image := _extract_image(response)) is None:
            raise RuntimeError("Expected image in response, but got none")

        resource = generated_image.resource
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

        if revised_prompt := generated_image.revised_prompt:
            await consumer.add_attachment(
                Attachment(title="Revised prompt", data=revised_prompt)
            )

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
        deployment: AdapterDeployment[ImagenDeployment],
        config: UpstreamConfig,
    ) -> "ImagenChatCompletionAdapter":
        client = await config.get_genai_client()
        return cls(file_storage, client, deployment)


def _prepare_generation_config(
    params: ModelParameters, config: ImagenConfig | None
) -> GenerateImagesConfigDict:
    return (config or ImagenConfig()).to_config_dict(params.seed)


class GeneratedImage(BaseModel):
    resource: Resource
    revised_prompt: str | None


def _extract_image(response: GenerateImagesResponse) -> GeneratedImage | None:
    images = response.generated_images
    if images is None or len(images) == 0:
        return None

    if len(images) > 1:
        log.warning(
            f"Expected to receive 1 generated image, but got {len(images)}. Only the first is taken into account."
        )

    generated_image = images[0]
    revised_prompt = generated_image.enhanced_prompt

    if reason := generated_image.rai_filtered_reason:
        raise InvalidRequestError(code="content_filter", message=reason)

    if (image := generated_image.image) is None:
        return None

    if (image_data := image.image_bytes) is None:
        return None

    if (pil_image := image._pil_image) is None:
        return None

    media_type = _get_image_type(pil_image)

    resource = Resource(type=media_type, data=image_data)

    return GeneratedImage(resource=resource, revised_prompt=revised_prompt)


def _get_image_type(image: PIL_Image.Image) -> str:
    match image.format:
        case "JPEG":
            return "image/jpeg"
        case "PNG":
            return "image/png"
        case _:
            raise ValueError(f"Unknown image format: {image.format}")
