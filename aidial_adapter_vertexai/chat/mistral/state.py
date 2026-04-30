import json
from dataclasses import dataclass
from typing import Any

from mistralai.gcp.client.models import (
    FunctionCall,
    ToolCall,
)

from aidial_adapter_vertexai.chat.errors import ValidationError


@dataclass
class _ToolCallState:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def to_tool_call(self) -> ToolCall:
        if self.name is None:
            raise ValidationError(
                "Invalid streamed tool call: function name is missing"
            )
        return ToolCall(
            id=self.id or "null",
            index=self.index,
            function=FunctionCall(
                name=self.name,
                arguments=_to_json_object_or_string(self.arguments),
            ),
        )


def _to_json_object_or_string(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    try:
        return json.loads(value)
    except ValueError:
        return value
