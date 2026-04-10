import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from logging import DEBUG

from aidial_sdk.chat_completion import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from google.genai.client import Client as GenAIClient
from google.genai.types import (
    ContentListUnion,
    ContentUnion,
    EmbedContentConfig,
    EmbedContentResponse,
    Part,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.deployments import GenAIEmbeddingDeployment
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs,
)
from aidial_adapter_vertexai.dial_api.resource import AttachmentResource
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.types import (
    Embedding,
    make_embeddings_response,
    vector_to_embedding,
)
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log


@dataclass
class EmbeddingsInput:
    input: ContentListUnion
    config: EmbedContentConfig | None = None


async def compute_embeddings(
    client: GenAIClient,
    model_id: str,
    base64_encode: bool,
    inputs: list[EmbeddingsInput],
) -> tuple[list[Embedding], int]:
    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short({"inputs": inputs})
        log.debug(f"request: {msg}")

    # NOTE: it's possible to batch the inputs (using the config as a clustering key)
    # and therefore make less requests to the upstream.
    # However, there are a few issues with it:
    # (1) the batch request may fail with 413, so
    #     we have to limit the size of the batch somehow;
    # (2) batching logic is outside of the user control.
    tasks = [
        client.aio.models.embed_content(
            model=model_id, contents=i.input, config=i.config
        )
        for i in inputs
    ]

    responses: list[EmbedContentResponse] = await asyncio.gather(*tasks)

    if log.isEnabledFor(DEBUG):
        log.debug(f"responses: {json_dumps_short(responses)}")

    embeddings: list[Embedding] = []
    tokens = 0

    for response in responses:
        for embedding in response.embeddings or []:
            if values := embedding.values:
                embeddings.append(vector_to_embedding(base64_encode, values))

                if (stat := embedding.statistics) and (
                    count := stat.token_count
                ):
                    tokens += int(count)

    return embeddings, tokens


async def get_embedding_requests(
    storage: FileStorage | None,
    request: EmbeddingsRequest,
    task_type: str | None,
    dimensions: int | None,
) -> AsyncIterator[EmbeddingsInput]:
    def create_config(*, title: str | None = None) -> EmbedContentConfig:
        return EmbedContentConfig(
            title=title,
            task_type=task_type,
            output_dimensionality=dimensions,
        )

    async def download_attachment(attachment: Attachment) -> Part:
        attachment_resource = AttachmentResource(attachment=attachment)
        resource = await attachment_resource.download(storage)
        return Part.from_bytes(data=resource.data, mime_type=resource.type)

    async def download_text_or_attachment(
        input: str | Attachment,
    ) -> ContentUnion:
        if isinstance(input, str):
            return input
        else:
            return await download_attachment(input)

    async def on_mixed_one(input: str | Attachment):
        return EmbeddingsInput(
            input=await download_text_or_attachment(input),
            config=create_config(),
        )

    async def on_mixed_many(inputs: list[str | Attachment]) -> EmbeddingsInput:
        if not inputs:
            raise EMPTY_INPUT_LIST_ERROR

        title = None
        parts = [await download_text_or_attachment(input) for input in inputs]

        if len(inputs) > 1 and isinstance(inputs[0], str):
            title, parts = inputs[0], parts[1:]

        config = create_config(title=title)
        return EmbeddingsInput(parts, config=config)

    return collect_embedding_inputs(
        request,
        on_text=on_mixed_one,
        on_attachment=on_mixed_one,
        on_mixed=on_mixed_many,
    )


@dataclass
class GenAIEmbeddingsAdapter(EmbeddingsAdapter):
    deployment: AdapterDeployment[GenAIEmbeddingDeployment]
    client: GenAIClient
    storage: FileStorage | None

    @classmethod
    async def create(
        cls,
        storage: FileStorage | None,
        deployment: AdapterDeployment[GenAIEmbeddingDeployment],
        config: UpstreamConfig,
    ) -> "EmbeddingsAdapter":
        client = await config.get_genai_client()
        return cls(deployment=deployment, client=client, storage=storage)

    @property
    def model_id(self) -> str:
        return self.deployment.upstream_deployment_id

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        if (
            request.custom_fields is not None
            and request.custom_fields.instruction is not None
        ):
            raise ValidationError("Instruction prompt is not supported")

        task_type: str | None = None
        if request.custom_fields is not None:
            task_type = request.custom_fields.type

        input_iter = await get_embedding_requests(
            self.storage, request, task_type, request.dimensions
        )
        inputs: list[EmbeddingsInput] = [input async for input in input_iter]

        base64_encode = request.encoding_format == "base64"

        embeddings, tokens = await compute_embeddings(
            self.client, self.model_id, base64_encode, inputs
        )

        return make_embeddings_response(
            model=self.deployment.upstream_deployment_id,
            embeddings=embeddings,
            tokens=tokens,
        )
