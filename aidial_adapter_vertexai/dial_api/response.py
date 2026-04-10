from typing import Literal, Self

from pydantic import BaseModel


class Capabilities(BaseModel):
    chat_completion: bool = False
    completion: bool = False
    embeddings: bool = False
    fine_tune: bool = False
    inference: bool = False


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    capabilities: Capabilities = Capabilities()
    id: str

    @classmethod
    def chat_completions(cls, id: str) -> Self:
        return cls(id=id, capabilities=Capabilities(chat_completion=True))

    @classmethod
    def embeddings(cls, id: str) -> Self:
        return cls(id=id, capabilities=Capabilities(embeddings=True))


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
