from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional

from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FinishReason,
    Response,
    Stage,
)

from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage


class Consumer(ABC):
    @abstractmethod
    async def append_content(self, content: str): ...

    @abstractmethod
    async def create_function_call(self, name: str, arguments: str | None): ...

    @abstractmethod
    async def create_tool_call(
        self, id: str, name: str, arguments: str | None
    ): ...

    @abstractmethod
    async def add_attachment(self, attachment: Attachment): ...

    @abstractmethod
    async def set_usage(self, usage: TokenUsage): ...

    @abstractmethod
    async def set_finish_reason(self, finish_reason: FinishReason): ...

    @abstractmethod
    def get_finish_reason(self) -> FinishReason | None: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    async def create_stage(self, name: str) -> Stage: ...

    @property
    @abstractmethod
    def has_function_call(self) -> bool: ...


class ChoiceConsumer(Consumer):
    response: Response
    _choice: Optional[Choice]
    usage: TokenUsage
    finish_reason: Optional[FinishReason]

    empty: bool
    """
    Whether the consumer has sent something to the choice or not.
    """

    def __init__(self, response: Response):
        self.response = response
        self._choice = None
        self.empty = True
        self.usage = TokenUsage()
        self.finish_reason = None

    @property
    def choice(self) -> Choice:
        if self._choice is None:
            # Delay opening a choice to the very last moment
            # so as to give opportunity for exceptions to bubble up to
            # the level of HTTP response (instead of error objects in a stream).
            choice = self._choice = self.response.create_choice()
            choice.open()
            return choice
        else:
            return self._choice

    @property
    def choice_idx(self) -> int | None:
        if self._choice is None:
            return None
        return self._choice.index

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc is None and self._choice is not None:
            self._choice.close(self.finish_reason)
        return False

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
        self.choice.add_attachment(attachment)

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

    def get_finish_reason(self) -> FinishReason | None:
        return self.finish_reason

    async def create_stage(self, name) -> Stage:
        return self.choice.create_stage(name)

    @property
    def has_function_call(self) -> bool:
        return self.choice.has_function_call
