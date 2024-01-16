from typing import Callable, Coroutine, List, Optional

from aidial_sdk.chat_completion import Attachment, FinishReason

from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage

ContentCallback = Callable[[str], Coroutine[None, str, None]]


class CollectConsumer(Consumer):
    usage: TokenUsage
    content: str
    attachments: List[Attachment]
    finish_reason: Optional[FinishReason]

    on_content: Optional[ContentCallback]

    def __init__(self, on_content: Optional[ContentCallback] = None):
        self.usage = TokenUsage()
        self.content = ""
        self.attachments = []
        self.finish_reason = None

        self.on_content = on_content

    async def append_content(self, content: str):
        if self.on_content:
            await self.on_content(content)
        self.content += content

    async def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage

    async def set_finish_reason(self, finish_reason: Optional[FinishReason]):
        if self.finish_reason is None:
            self.finish_reason = finish_reason
        else:
            assert (
                self.finish_reason == finish_reason
            ), "finish_reason was set twice with different values"
