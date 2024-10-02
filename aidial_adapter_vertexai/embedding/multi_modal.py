from logging import DEBUG
from typing import AsyncIterator, Callable, List, Tuple

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.attachments import (
    download_with_content_type,
)
from aidial_adapter_vertexai.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs,
)
from aidial_adapter_vertexai.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_vertexai.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_vertexai.embedding.types import (
    Embedding,
    make_embeddings_response,
    vector_to_embedding,
)
from aidial_adapter_vertexai.utils.concurrency import gather_sync
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.vertex_ai import get_multi_modal_embedding_model

# See the documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api

SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"]


class ModelRequest(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    image: Image | None = None
    contextual_text: str | None = None

    def count_input_tokens(self) -> int:
        # The model doesn't report the number of input tokens.
        # However, one could count it oneself:
        # https://cloud.google.com/vertex-ai/generative-ai/pricing#embedding-models
        # As of 29 Jul 2024, one image costs as much as 500 text input characters
        ret = len(self.contextual_text or "")
        if self.image:
            ret += 500
        return ret

    def extract_embeddings(
        self, response: MultiModalEmbeddingResponse
    ) -> Tuple[List[float], int]:

        vector: List[float] | None = None
        if self.image:
            vector = response.image_embedding
        else:
            vector = response.text_embedding

        if vector is None:
            raise ValueError("No embeddings returned")

        return vector, self.count_input_tokens()


def compute_embeddings(
    request: ModelRequest,
    model: MultiModalEmbeddingModel,
    base64_encode: bool,
    dimensions: int | None,
) -> Tuple[Embedding, int]:

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(
            {
                "image": request.image,
                "contextual_text": request.contextual_text,
                "dimension": dimensions,
            }
        )
        log.debug(f"request: {msg}")

    response: MultiModalEmbeddingResponse = model.get_embeddings(
        image=request.image,
        contextual_text=request.contextual_text,
        dimension=dimensions,
    )

    if log.isEnabledFor(DEBUG):
        msg = json_dumps_short(response)
        log.debug(f"response: {msg}")

    vec, tokens = request.extract_embeddings(response)

    return vector_to_embedding(base64_encode, vec), tokens


def validate_request(request: EmbeddingsRequest) -> None:
    if request.custom_fields is not None:
        if request.custom_fields.instruction is not None:
            raise ValidationError("Instruction prompt is not supported")

        if request.custom_fields.type is not None:
            raise ValidationError(
                "The embedding model does not support embedding types"
            )


async def get_requests(
    request: EmbeddingsRequest,
    storage: FileStorage | None,
) -> AsyncIterator[ModelRequest]:
    async def download_image(attachment: Attachment) -> Image:
        data = await download_with_content_type(
            SUPPORTED_IMAGE_TYPES, storage, attachment
        )
        return Image(image_bytes=data)

    async def on_text(text: str):
        return ModelRequest(contextual_text=text)

    async def on_attachment(attachment: Attachment):
        return ModelRequest(image=await download_image(attachment))

    async def on_mixed(inputs: List[str | Attachment]) -> ModelRequest:
        if len(inputs) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(inputs) == 1:
            if isinstance(inputs[0], str):
                return await on_text(inputs[0])
            else:
                return await on_attachment(inputs[0])
        elif len(inputs) == 2:
            if isinstance(inputs[0], str) and isinstance(inputs[1], Attachment):
                return ModelRequest(
                    contextual_text=inputs[0],
                    image=await download_image(inputs[1]),
                )
            elif isinstance(inputs[0], Attachment) and isinstance(
                inputs[1], str
            ):
                return ModelRequest(
                    contextual_text=inputs[1],
                    image=await download_image(inputs[0]),
                )
            else:
                raise ValidationError(
                    "The first element of a custom_input list element must be a string and the second element must be an image attachment or vice versa"
                )
        else:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list"
            )

    return collect_embedding_inputs(
        request,
        on_text=on_text,
        on_attachment=on_attachment,
        on_mixed=on_mixed,
    )


class MultiModalEmbeddingsAdapter(EmbeddingsAdapter):
    model_id: str
    model: MultiModalEmbeddingModel
    api_key: str
    storage: FileStorage | None

    @classmethod
    async def create(
        cls,
        model_id: str,
        project_id: str,
        location: str,
        api_key: str,
    ) -> "EmbeddingsAdapter":
        storage = create_file_storage(api_key)
        model = await get_multi_modal_embedding_model(model_id)
        return cls(
            model_id=model_id, model=model, api_key=api_key, storage=storage
        )

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:

        validate_request(request)

        base64_encode = request.encoding_format == "base64"

        # NOTE: The model doesn't support batched inputs
        tasks: List[Callable[[], Tuple[Embedding, int]]] = []
        async for sub_request in await get_requests(request, self.storage):
            tasks.append(
                lambda sub_req=sub_request: compute_embeddings(
                    sub_req,
                    self.model,
                    base64_encode=base64_encode,
                    dimensions=request.dimensions,
                )
            )

        embeddings: List[Embedding] = []
        total_tokens = 0

        for embedding, tokens in await gather_sync(tasks):
            embeddings.append(embedding)
            total_tokens += tokens

        return make_embeddings_response(
            model=self.model_id,
            embeddings=embeddings,
            tokens=total_tokens,
        )
