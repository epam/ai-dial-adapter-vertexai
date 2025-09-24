from aiocache import cached
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from aidial_adapter_vertexai.utils.concurrency import make_single_thread_async


@cached()
async def get_text_embedding_model(model_id: str) -> TextEmbeddingModel:
    return await make_single_thread_async(
        TextEmbeddingModel.from_pretrained, model_id
    )


@cached()
async def get_multi_modal_embedding_model(
    model_id: str,
) -> MultiModalEmbeddingModel:
    return await make_single_thread_async(
        MultiModalEmbeddingModel.from_pretrained, model_id
    )
