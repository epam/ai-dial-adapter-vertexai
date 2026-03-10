from typing import Any, Dict

from pydantic import BaseModel


class ExtraForbidModel(BaseModel, extra="forbid"):
    pass


class ExtraAllowModel(BaseModel, extra="allow"):
    @property
    def extra_fields(self) -> Dict[str, Any]:
        return self.model_extra or {}
