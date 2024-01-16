from abc import ABC, abstractmethod
from typing import Optional

from aidial_sdk.chat_completion import Attachment, Choice, FinishReason

from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage


class Consumer(ABC):
    @abstractmethod
    async def append_content(self, content: str):
        pass

    @abstractmethod
    async def add_attachment(self, attachment: Attachment):
        pass

    @abstractmethod
    async def set_usage(self, usage: TokenUsage):
        pass

    @abstractmethod
    async def set_finish_reason(self, finish_reason: FinishReason):
        pass


class ChoiceConsumer(Consumer):
    choice: Choice
    usage: TokenUsage
    finish_reason: Optional[FinishReason]

    def __init__(self, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()
        self.finish_reason = None

    async def append_content(self, content: str):
        self.choice.append_content(content)

    async def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(
            type=attachment.type,
            title=attachment.title,
            data=attachment.data,
            url=attachment.url,
            reference_url=attachment.reference_url,
            reference_type=attachment.reference_type,
        )

    async def set_usage(self, usage: TokenUsage):
        self.usage = usage

    async def set_finish_reason(self, finish_reason: FinishReason):
        if self.finish_reason is None:
            self.finish_reason = finish_reason
        else:
            assert (
                self.finish_reason == finish_reason
            ), "finish_reason was set twice with different values"
