from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional, Self, Set, Sized, Tuple

from aidial_sdk.exceptions import ContextLengthExceededError
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import (
    InvalidRequestError,
    TruncatePromptSystemAndLastUserError,
)
from pydantic import BaseModel


class TruncatePromptError(ABC, BaseModel):
    @abstractmethod
    def to_dial_exception(self) -> DialException:
        pass

    def print(self) -> str:
        return self.to_dial_exception().message


class InconsistentLimitsError(TruncatePromptError):
    user_limit: int
    model_limit: int

    def to_dial_exception(self) -> DialException:
        return InvalidRequestError(
            f"The request maximum prompt tokens is {self.user_limit}. "
            f"However, the model's maximum context length is {self.model_limit} tokens."
        )


class ModelLimitOverflowError(TruncatePromptError):
    model_limit: int
    token_count: int

    def to_dial_exception(self) -> DialException:
        return ContextLengthExceededError(self.model_limit, self.token_count)


class UserLimitOverflowError(TruncatePromptError):
    user_limit: int
    token_count: int

    def to_dial_exception(self) -> DialException:
        return TruncatePromptSystemAndLastUserError(
            self.user_limit, self.token_count
        )


def _partition_indexer(chunks: List[int]) -> Callable[[int], List[int]]:
    """Returns a function that maps an index to indices of its partition.
    >>> [_partition_indexer([2, 3])(i) for i in range(5)]
    [[0, 1], [0, 1], [2, 3, 4], [2, 3, 4], [2, 3, 4]]
    """
    mapping: dict[int, List[int]] = {}
    offset = 0
    for size in chunks:
        chunk = list(range(offset, offset + size))
        for idx in range(size):
            mapping[offset + idx] = chunk
        offset += size

    return mapping.__getitem__


DiscardedMessages = List[int]


class TruncatablePrompt(ABC, Sized):

    @abstractmethod
    def is_required_message(self, index: int) -> bool:
        """
        Returns True if the message at the given index is required,
        meaning that the prompt truncation algorithm should not
        remove such a message.

        Typically it's all system messages and the last message in the conversation.
        """
        ...

    @abstractmethod
    def partition_messages(self) -> List[int]:
        """
        Returns a list of sizes of contiguous non-overlapping sequences of messages
        which together represent partition of the list of messages.

        Therefore, the sum of the sizes must be equal to the number of messages.

        Each partition is either preserved or discarded by the prompt truncation algorithm *as a whole*.

        Typical partition are:
        * the trivial partition that turns each message into a single-element block of messages
        * the turn-based partition that consolidates user-bot turns into two-element blocks of messages.
            This is useful for models that require the conversation to be composed of
            a whole number of user-bot turns followed by a user query.
        """
        ...

    @abstractmethod
    def select(self, indices: Set[int]) -> Self:
        """
        Return a new prompt composed of the messages with the given indices.
        """
        ...

    def omit(self, indices: Set[int]) -> Self:
        return self.select(set(range(len(self))) - indices)

    async def truncate(
        self,
        *,
        tokenizer: Callable[[Self], Awaitable[int]],
        model_limit: Optional[int] = None,
        user_limit: Optional[int] = None,
    ) -> Tuple[DiscardedMessages, Self]:
        """
        Returns a list of indices of discarded messages and
        the truncated prompt that doesn't include the discarded messages and fits into the given user limit.

        The list of discarded messages is a prefix of the list of non-required messages and
        its length is as minimal as possible to fit the truncated prompt into the given user limit.

        Parameters:
        * The tokenizer computes number of tokens in the given prompt.
        * The model limit is the intrinsic context limit on the number of input tokes for the given model.
        * The user limit (aka max_prompt_tokens) defines the number of tokens that the resulting truncated prompt must fit in.

        Throws a DIAL exception when the truncation satisfying the given limits is impossible.
        """

        result = await self.compute_discarded_messages(
            tokenizer=tokenizer,
            model_limit=model_limit,
            user_limit=user_limit,
        )

        if isinstance(result, TruncatePromptError):
            raise result.to_dial_exception()

        return (list(result), self.omit(set(result)))

    async def compute_discarded_messages(
        self,
        *,
        tokenizer: Callable[[Self], Awaitable[int]],
        model_limit: Optional[int],
        user_limit: Optional[int],
    ) -> DiscardedMessages | TruncatePromptError:
        if (
            user_limit is not None
            and model_limit is not None
            and user_limit > model_limit
        ):
            return InconsistentLimitsError(
                user_limit=user_limit, model_limit=model_limit
            )

        if user_limit is None:
            if model_limit is None:
                return []

            token_count = await tokenizer(self)
            if token_count <= model_limit:
                return []

            return ModelLimitOverflowError(
                model_limit=model_limit, token_count=token_count
            )

        partition_sizes = self.partition_messages()
        if sum(partition_sizes) != len(self):
            raise ValueError(
                "Partition sizes must add up to the number of messages."
            )

        async def _tokenize_selected(indices: Set[int]) -> int:
            return await tokenizer(self.select(indices))

        get_partition_indices = _partition_indexer(partition_sizes)

        n = len(self)
        kept_indices: Set[int] = {
            j
            for i in range(n)
            for j in get_partition_indices(i)
            if self.is_required_message(i)
        }

        token_count = await _tokenize_selected(kept_indices)
        if token_count > user_limit:
            return UserLimitOverflowError(
                user_limit=user_limit, token_count=token_count
            )

        for idx in reversed(range(n)):
            if idx in kept_indices:
                continue

            chunk_indices = get_partition_indices(idx)
            new_token_count = await _tokenize_selected(
                {*kept_indices, *chunk_indices}
            )
            if new_token_count > user_limit:
                break

            kept_indices.update(chunk_indices)

        all_indices = set(range(n))
        return sorted(list(all_indices - kept_indices))
