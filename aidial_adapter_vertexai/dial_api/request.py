from typing import List, Mapping, Optional, assert_never

from aidial_sdk.chat_completion import (
    Attachment,
    Message,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    Request,
)
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.errors import ValidationError


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


def collect_text_content(
    content: str | List[MessageContentPart] | None, delimiter: str = "\n\n"
) -> str:
    match content:
        case None:
            return ""
        case str():
            return content
        case list():
            texts: List[str] = []
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        texts.append(text)
                    case MessageContentImagePart():
                        raise ValidationError(
                            "Can't extract text from an image content part"
                        )
                    case _:
                        assert_never(part)
            return delimiter.join(texts)
        case _:
            assert_never(content)
