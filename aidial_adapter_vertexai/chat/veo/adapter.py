import asyncio
from logging import DEBUG
from typing import List, Optional

from aidial_sdk.chat_completion import Attachment, Message
from aidial_sdk.exceptions import InvalidRequestError
from google.genai.client import Client as GenAIClient
from google.genai.types import GenerateVideosConfigDict, GenerateVideosOperation
from pydantic import BaseModel
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.chat.veo.configuration import VeoConfig
from aidial_adapter_vertexai.deployments import VeoDeployment
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

VeoPrompt = str


class VeoChatCompletionAdapter(ChatCompletionAdapter[VeoPrompt]):
    file_storage: FileStorage | None
    client: GenAIClient
    deployment: AdapterDeployment[VeoDeployment]

    def __init__(
        self,
        file_storage: FileStorage | None,
        client: GenAIClient,
        deployment: AdapterDeployment[VeoDeployment],
    ):
        self.file_storage = file_storage
        self.client = client
        self.deployment = deployment

    @property
    def model_id(self) -> str:
        return self.deployment.upstream_deployment_id

    async def configuration(self) -> type[VeoConfig]:
        return VeoConfig

    @override
    async def parse_prompt(
        self,
        params: ModelParameters,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> VeoPrompt:
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
        self, prompt: VeoPrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[VeoPrompt]:
        return TruncatedPrompt(discarded_messages=[], prompt=prompt)

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: VeoPrompt
    ) -> None:
        configuration = params.parse_configuration(await self.configuration())
        config = _prepare_generation_config(params, configuration)

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"model": self.model_id, "prompt": prompt, "config": config}
            )
            log.debug(f"request: {msg}")

        with Timer("predict timing: {time}", log.debug):
            operation = await self.client.aio.models.generate_videos(
                model=self.model_id, prompt=prompt, config=config
            )
            poll_interval = 3
            while not operation.done:
                log.debug(f"Sleeping {poll_interval} seconds")
                await asyncio.sleep(poll_interval)
                operation = await self.client.aio.operations.get(operation)
            response = operation.response

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(response)}")

        if (generated_video := _extract_video(operation)) is None:
            raise RuntimeError("Expected video in response, but got none")

        resource = generated_video.resource
        attachment = Attachment(
            title="Video",
            type=resource.type,
            data=resource.data_base64,
        )
        if self.file_storage is not None:
            with Timer("upload to file storage: {time}", log.debug):
                filename = "videos/" + compute_hash_digest(resource.data)
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
    async def count_prompt_tokens(self, prompt: VeoPrompt) -> int:
        return 0

    @override
    async def count_completion_tokens(self, string: str) -> int:
        return 1

    @classmethod
    async def create(
        cls,
        file_storage: Optional[FileStorage],
        deployment: AdapterDeployment[VeoDeployment],
        config: UpstreamConfig,
    ) -> "VeoChatCompletionAdapter":
        client = await config.get_genai_client()
        return cls(file_storage, client, deployment)


def _prepare_generation_config(
    params: ModelParameters, config: VeoConfig | None
) -> GenerateVideosConfigDict:
    return (config or VeoConfig()).to_config_dict(params.seed)


class GeneratedVideo(BaseModel):
    resource: Resource


def _extract_video(response: GenerateVideosOperation) -> GeneratedVideo | None:
    result = response.response
    if result is None:
        return None

    videos = result.generated_videos
    if videos is None or len(videos) == 0:
        return None

    if len(videos) > 1:
        log.warning(
            f"Expected to receive 1 generated video, but got {len(videos)}. Only the first is taken into account."
        )

    if result.rai_media_filtered_count == 0:
        log.warning("The video was filtered due to RAI policies.")

    if result.rai_media_filtered_reasons:
        raise InvalidRequestError(
            code="content_filter",
            message=", ".join(result.rai_media_filtered_reasons),
        )

    generated_video = videos[0]

    if (video := generated_video.video) is None:
        return None

    if (video_data := video.video_bytes) is None:
        return None

    video_type = video.mime_type or "video/mp4"
    resource = Resource(type=video_type, data=video_data)

    return GeneratedVideo(resource=resource)
