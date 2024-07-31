from typing import List, Literal

from pydantic import BaseModel


class ModelObject(BaseModel):
    id: str
    object: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]
