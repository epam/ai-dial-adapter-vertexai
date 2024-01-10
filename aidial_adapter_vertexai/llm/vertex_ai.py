from functools import cache

import vertexai
from aiocache import cached
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel

from aidial_adapter_vertexai.llm.vertex_ai_chat import VertexAIChat
from aidial_adapter_vertexai.utils.concurrency import make_async


# TODO: For now assume that there will be only one project and location.
# We need to fix it otherwise.
@cached()
async def init_vertex_ai(project_id: str, location: str) -> None:
    await make_async(
        lambda _: vertexai.init(project=project_id, location=location), ()
    )


@cache
def get_vertex_ai_chat(
    model_id: str, project_id: str, location: str
) -> VertexAIChat:
    return VertexAIChat.create(model_id, project_id, location)


@cached()
async def get_gemini_model(model_id: str) -> GenerativeModel:
    return await make_async(GenerativeModel, model_id)


@cached()
async def get_embedding_model(model_id: str) -> TextEmbeddingModel:
    return await make_async(TextEmbeddingModel.from_pretrained, model_id)
