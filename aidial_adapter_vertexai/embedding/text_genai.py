import asyncio
from dataclasses import dataclass
from logging import DEBUG
from typing import List, Optional, Tuple

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from google.genai.client import Client as GenAIClient
from google.genai.types import (
    ContentListUnion,
    EmbedContentConfig,
    EmbedContentResponse,
)

from aidial_adapter_vertexai.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.deployments import TextEmbeddingDeployment
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs_without_attachments,
)
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.types import (
    Embedding,
    make_embeddings_response,
    vector_to_embedding,
)
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log

Input = Tuple[ContentListUnion, EmbedContentConfig | None]


async def compute_embeddings(
    client: GenAIClient,
    model_id: str,
    base64_encode: bool,
    inputs: List[Input],
) -> Tuple[List[Embedding], int]:

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short({"inputs": inputs})
        log.debug(f"request: {msg}")

    tasks = [
        client.aio.models.embed_content(
            model=model_id, contents=contents, config=config
        )
        for (contents, config) in inputs
    ]

    responses: List[EmbedContentResponse] = await asyncio.gather(*tasks)

    if log.isEnabledFor(DEBUG):
        log.debug(f"responses: {json_dumps_short(responses)}")

    embeddings: List[Embedding] = []
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


async def get_embedding_inputs(
    request: EmbeddingsRequest,
    task_type: Optional[str],
    dimensions: int | None,
) -> List[Input]:
    async def on_texts(texts: List[str]) -> Input:
        if len(texts) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(texts) == 1:
            title, text = None, texts[0]
        elif len(texts) == 2:
            title, text = texts
            if task_type != "RETRIEVAL_DOCUMENT":
                raise ValidationError(
                    "The model does not support inputs with titles "
                    "unless the type is RETRIEVAL_DOCUMENT"
                )
        else:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list - one for title and one for text."
            )

        return text, EmbedContentConfig(
            title=title,
            task_type=task_type,
            output_dimensionality=dimensions,
        )

    iterator = collect_embedding_inputs_without_attachments(
        request, on_texts=on_texts
    )

    return [input async for input in iterator]


@dataclass
class TextEmbeddingsAdapter(EmbeddingsAdapter):
    deployment: AdapterDeployment[TextEmbeddingDeployment]
    client: GenAIClient

    @classmethod
    async def create(
        cls,
        deployment: AdapterDeployment[TextEmbeddingDeployment],
        config: UpstreamConfig,
    ) -> "EmbeddingsAdapter":
        return cls(
            deployment=deployment, client=await config.get_genai_client()
        )

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

        inputs: List[Input] = await get_embedding_inputs(
            request, task_type, request.dimensions
        )

        base64_encode = request.encoding_format == "base64"

        embeddings, tokens = await compute_embeddings(
            self.client, self.model_id, base64_encode, inputs
        )

        return make_embeddings_response(
            model=self.deployment.upstream_deployment_id,
            embeddings=embeddings,
            tokens=tokens,
        )
