from typing import Any, Dict

from pydantic.v1 import BaseModel


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"


class ExtraAllowModel(BaseModel):
    class Config:
        extra = "allow"

    @property
    def extra_fields(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self.__fields__
        }
