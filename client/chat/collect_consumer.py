from typing import Callable, Coroutine, List, Optional

from aidial_sdk.chat_completion import (
    Attachment,
    CustomContent,
    FinishReason,
    FunctionCall,
    Message,
    Role,
    ToolCall,
)

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage

ContentCallback = Callable[[str], Coroutine[None, str, None]]


class CollectConsumer(Consumer):
    usage: TokenUsage
    attachments: List[Attachment]
    finish_reason: Optional[FinishReason]

    on_content: Optional[ContentCallback]

    content: str
    tool_calls: List[ToolCall] | None
    function_call: FunctionCall | None

    def __init__(self, on_content: Optional[ContentCallback] = None):
        self.usage = TokenUsage()
        self.attachments = []
        self.finish_reason = None
        self.on_content = on_content

        self.content = ""
        self.tool_calls = None
        self.function_call = None

    def to_message(self) -> Message:
        return Message(
            role=Role.ASSISTANT,
            content=self.content,
            tool_calls=self.tool_calls,
            function_call=self.function_call,
            custom_content=CustomContent(attachments=self.attachments),
        )

    def is_empty(self) -> bool:
        return False

    async def create_function_call(self, name: str, arguments: str | None):
        await self.set_finish_reason(FinishReason.FUNCTION_CALL)
        self.function_call = FunctionCall(name=name, arguments=arguments or "")

    async def create_tool_call(self, id: str, name: str, arguments: str | None):
        await self.set_finish_reason(FinishReason.TOOL_CALLS)
        if self.tool_calls is None:
            self.tool_calls = []
        function = FunctionCall(name=name, arguments=arguments or "")
        self.tool_calls.append(
            ToolCall(index=None, id=id, type="function", function=function)
        )

    async def append_content(self, content: str):
        if self.on_content:
            await self.on_content(content)
        self.content += content

    async def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage

    async def set_finish_reason(self, finish_reason: FinishReason):
        if finish_reason == FinishReason.STOP and self.finish_reason in [
            FinishReason.FUNCTION_CALL,
            FinishReason.TOOL_CALLS,
        ]:
            return

        if (
            self.finish_reason is not None
            and self.finish_reason != finish_reason
        ):
            raise RuntimeError(
                "finish_reason was set twice with different values: "
                f"{self.finish_reason}, {finish_reason}"
            )

        self.finish_reason = finish_reason
