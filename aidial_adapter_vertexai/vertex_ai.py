from aiocache import cached
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import (
    ChatModel,
    CodeChatModel,
    TextEmbeddingModel,
)
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.vision_models import MultiModalEmbeddingModel

from aidial_adapter_vertexai.utils.concurrency import make_single_thread_async


@cached()
async def get_code_chat_model(model_id: str) -> CodeChatModel:
    # TODO: We're using single threaded async call, because
    # calling `from_pretrained` in different threads cause deadlock
    # https://github.com/googleapis/python-aiplatform/issues/4342
    # When this issue is resolved, we use just `make_async`
    return await make_single_thread_async(
        CodeChatModel.from_pretrained, model_id
    )


@cached()
async def get_chat_model(model_id: str) -> ChatModel:
    return await make_single_thread_async(ChatModel.from_pretrained, model_id)


@cached()
async def get_gemini_model(model_id: str) -> GenerativeModel:
    return await make_single_thread_async(GenerativeModel, model_id)


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


@cached()
async def get_image_generation_model(model_id: str) -> ImageGenerationModel:
    return await make_single_thread_async(
        ImageGenerationModel.from_pretrained, model_id
    )
