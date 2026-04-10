from typing import Any

from pydantic import BaseModel


class ExtraForbidModel(BaseModel, extra="forbid"):
    pass


class ExtraAllowModel(BaseModel, extra="allow"):
    @property
    def extra_fields(self) -> dict[str, Any]:
        return self.model_extra or {}
