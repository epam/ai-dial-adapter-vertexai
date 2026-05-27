from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType

from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FinishReason,
    Response,
    Stage,
)

from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage


class Consumer(AbstractContextManager, ABC):
    @abstractmethod
    def fork(self) -> Consumer: ...

    @property
    @abstractmethod
    def choice(self) -> Choice: ...

    @abstractmethod
    def get_response(self) -> Response: ...

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
    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int: ...

    @abstractmethod
    async def set_usage(self, usage: TokenUsage): ...

    @abstractmethod
    async def set_state(self, state: dict): ...

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


@dataclass
class _ResponseState:
    usage: TokenUsage = field(default_factory=TokenUsage)
    empty: bool = True
    """
    Whether the consumer has sent something to any of choices or not.
    """


@dataclass
class ChoiceConsumer(Consumer):
    response: Response
    response_state: _ResponseState = field(default_factory=_ResponseState)

    choice_: Choice | None = None
    finish_reason: FinishReason | None = None
    citations: dict[int, tuple[int, Attachment | None]] = field(
        default_factory=dict
    )
    """
    Mapping from the document ID to a tuple of (1-based display index, attachment).
    """

    def fork(self) -> Consumer:
        return ChoiceConsumer(
            response=self.response,
            response_state=self.response_state,
        )

    def get_response(self) -> Response:
        return self.response

    @property
    def choice(self) -> Choice:
        if self.choice_ is None:
            # Delay opening a choice to the very last moment
            # so as to give opportunity for exceptions to bubble up to
            # the level of HTTP response (instead of error objects in a stream).
            choice = self.choice_ = self.response.create_choice()
            choice.open()
            return choice
        else:
            return self.choice_

    @property
    def choice_idx(self) -> int | None:
        if self.choice_ is None:
            return None
        return self.choice_.index

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc is None and self.choice_ is not None:
            self.choice_.close(self.finish_reason)
        return False

    def is_empty(self) -> bool:
        return self.response_state.empty

    def become_non_empty(self):
        self.response_state.empty = False

    async def create_function_call(self, name: str, arguments: str | None):
        self.become_non_empty()
        await self.set_finish_reason(FinishReason.FUNCTION_CALL)
        self.choice.create_function_call(name, arguments)

    async def create_tool_call(self, id: str, name: str, arguments: str | None):
        self.become_non_empty()
        await self.set_finish_reason(FinishReason.TOOL_CALLS)
        self.choice.create_function_tool_call(id, name, arguments)

    async def append_content(self, content: str):
        if content != "":
            self.become_non_empty()
        self.choice.append_content(content)

    async def add_attachment(self, attachment: Attachment):
        self.become_non_empty()
        self.choice.add_attachment(attachment)

    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int:
        if document_id in self.citations:
            return self.citations[document_id][0]

        display_index = len(self.citations) + 1
        self.citations[document_id] = (display_index, document)

        if document:
            document = document.copy()
            document.title = f"[{display_index}] {document.title or ''}".strip()
            document.reference_type = document.reference_type or document.type
            document.reference_url = document.reference_url or document.url
            await self.add_attachment(document)

        return display_index

    async def set_usage(self, usage: TokenUsage):
        # Avoiding the error from DIAL SDK:
        #   'Trying to set "usage" before generating all choices'
        if self.is_empty():
            await self.append_content("")
        self.response_state.usage = usage

    def get_usage(self) -> TokenUsage:
        return self.response_state.usage

    async def set_state(self, state: dict):
        if state:
            self.choice.set_state(state)

    async def set_finish_reason(self, finish_reason: FinishReason):
        if self.finish_reason in [
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
