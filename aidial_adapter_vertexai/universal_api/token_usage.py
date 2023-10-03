from typing import NotRequired, TypedDict

from pydantic import BaseModel


class TokenUsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: NotRequired[
        int
    ]  # None in case if nothing has been generated
    total_tokens: int


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> TokenUsageDict:
        ret: TokenUsageDict = {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }

        if self.completion_tokens != 0:
            ret["completion_tokens"] = self.completion_tokens

        return ret
