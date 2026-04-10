from vertexai.vision_models import MultiModalEmbeddingModel

from aidial_adapter_vertexai.utils.cache import cache
from aidial_adapter_vertexai.utils.concurrency import make_single_thread_async


@cache()
async def get_multi_modal_embedding_model(
    model_id: str,
) -> MultiModalEmbeddingModel:
    return await make_single_thread_async(
        MultiModalEmbeddingModel.from_pretrained, model_id
    )
