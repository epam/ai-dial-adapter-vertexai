import vertexai
from aiocache import cached
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import (
    ChatModel,
    CodeChatModel,
    TextEmbeddingModel,
)
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.vision_models import MultiModalEmbeddingModel


# TODO: For now assume that there will be only one project and location.
# We need to fix it otherwise.
@cached()
async def init_vertex_ai(project_id: str, location: str) -> None:
    vertexai.init(project=project_id, location=location)


@cached()
async def get_code_chat_model(model_id: str) -> CodeChatModel:
    return CodeChatModel.from_pretrained(model_id)


@cached()
async def get_chat_model(model_id: str) -> ChatModel:
    return ChatModel.from_pretrained(model_id)


@cached()
async def get_gemini_model(model_id: str) -> GenerativeModel:
    return GenerativeModel(model_id)


@cached()
async def get_text_embedding_model(model_id: str) -> TextEmbeddingModel:
    return TextEmbeddingModel.from_pretrained(model_id)


@cached()
async def get_multi_modal_embedding_model(
    model_id: str,
) -> MultiModalEmbeddingModel:
    return MultiModalEmbeddingModel.from_pretrained(model_id)


@cached()
async def get_image_generation_model(model_id: str) -> ImageGenerationModel:
    return ImageGenerationModel.from_pretrained(model_id)
