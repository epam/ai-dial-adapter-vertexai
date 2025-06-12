from pydantic.v1 import BaseModel


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"
