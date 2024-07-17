from typing import List, Mapping, Optional

from aidial_sdk.chat_completion import Attachment, Message, Request
from pydantic import BaseModel


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Mapping[int, float]] = None
    max_prompt_tokens: Optional[int] = None
    stream: bool = False

    @classmethod
    def create(cls, request: Request) -> "ModelParameters":
        stop = [request.stop] if isinstance(request.stop, str) else request.stop

        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            max_prompt_tokens=request.max_prompt_tokens,
            stream=request.stream,
        )


def get_attachments(message: Message) -> List[Attachment]:
    custom_content = message.custom_content
    if custom_content is None:
        return []
    return custom_content.attachments or []
