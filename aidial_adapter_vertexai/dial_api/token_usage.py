from aidial_sdk.chat_completion import Response
from aidial_sdk.chat_completion.chunks import PromptTokensDetails
from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cached_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def accumulate(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        if (
            self.prompt_cached_tokens is not None
            and other.prompt_cached_tokens is not None
        ):
            self.prompt_cached_tokens = (self.prompt_cached_tokens or 0) + (
                other.prompt_cached_tokens or 0
            )
        return self

    def set_response_usage(self, response: Response):
        details = None
        if self.prompt_cached_tokens:
            details = PromptTokensDetails(
                cached_tokens=self.prompt_cached_tokens
            )

        response.set_usage(
            self.prompt_tokens,
            self.completion_tokens,
            prompt_tokens_details=details,
        )
