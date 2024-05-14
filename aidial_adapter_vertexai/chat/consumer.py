from abc import ABC, abstractmethod
from typing import Optional

from aidial_sdk.chat_completion import Attachment, Choice, FinishReason

from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage


class Consumer(ABC):
    """
    Whether the consumer has sent something to the choice or not.
    """

    @abstractmethod
    async def append_content(self, content: str):
        pass

    @abstractmethod
    async def create_function_call(self, name: str, arguments: str | None):
        pass

    @abstractmethod
    async def create_tool_call(self, id: str, name: str, arguments: str | None):
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

    @abstractmethod
    def is_empty(self) -> bool:
        pass


class ChoiceConsumer(Consumer):
    choice: Choice
    usage: TokenUsage
    finish_reason: Optional[FinishReason]

    empty: bool
    """
    Whether the consumer has sent something to the choice or not.
    """

    def __init__(self, choice: Choice):
        self.empty = True
        self.choice = choice
        self.usage = TokenUsage()
        self.finish_reason = None

    def is_empty(self) -> bool:
        return self.empty

    async def create_function_call(self, name: str, arguments: str | None):
        self.empty = False
        await self.set_finish_reason(FinishReason.FUNCTION_CALL)
        self.choice.create_function_call(name, arguments)

    async def create_tool_call(self, id: str, name: str, arguments: str | None):
        self.empty = False
        await self.set_finish_reason(FinishReason.TOOL_CALLS)
        self.choice.create_function_tool_call(id, name, arguments)

    async def append_content(self, content: str):
        self.empty = self.empty and content == ""
        self.choice.append_content(content)

    async def add_attachment(self, attachment: Attachment):
        self.empty = False
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
