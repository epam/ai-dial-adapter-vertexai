from aidial_sdk.chat_completion import Response
from aidial_sdk.chat_completion.chunks import (
    CompletionTokensDetails,
    PromptTokensDetails,
)
from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cached_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    completion_reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def accumulate(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.prompt_cached_tokens += other.prompt_cached_tokens
        self.prompt_cache_write_tokens += other.prompt_cache_write_tokens
        self.completion_reasoning_tokens += other.completion_reasoning_tokens
        return self

    def set_response_usage(self, response: Response):
        prompt_details: PromptTokensDetails = {}
        if self.prompt_cached_tokens:
            prompt_details["cached_tokens"] = self.prompt_cached_tokens
        if self.prompt_cache_write_tokens:
            prompt_details["cache_write_tokens"] = (
                self.prompt_cache_write_tokens
            )

        completion_details: CompletionTokensDetails = {}
        if self.completion_reasoning_tokens:
            completion_details["reasoning_tokens"] = (
                self.completion_reasoning_tokens
            )

        response.set_usage(
            self.prompt_tokens,
            self.completion_tokens,
            prompt_tokens_details=prompt_details or None,
            completion_tokens_details=completion_details or None,
        )
