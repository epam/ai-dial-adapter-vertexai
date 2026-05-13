from dataclasses import dataclass

from mistralai.client.models import (
    FunctionCall,
    ToolCall,
)

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.utils.json import to_json_object_or_string


@dataclass
class ToolCallState:
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
                arguments=to_json_object_or_string(self.arguments),
            ),
        )
