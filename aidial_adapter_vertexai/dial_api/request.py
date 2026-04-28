from collections.abc import Mapping
from typing import (
    Literal,
    TypeGuard,
    TypeVar,
    assert_never,
)

from aidial_sdk.chat_completion import (
    Attachment,
    Message,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    ResponseFormat,
    Role,
)
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    MessageContentAudioPart,
    MessageContentRefusalPart,
    ReasoningEffort,
)
from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aidial_adapter_vertexai.chat.errors import ValidationError

_Model = TypeVar("_Model", bound=BaseModel)


class ModelParameters(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: Mapping[int, float] | None = None
    max_prompt_tokens: int | None = None
    stream: bool = False
    response_format: ResponseFormat | None = None
    seed: int | None = None
    configuration: dict | None = None
    reasoning_effort: ReasoningEffort | None = None

    @classmethod
    def create(cls, request: ChatCompletionRequest) -> "ModelParameters":
        stop = [request.stop] if isinstance(request.stop, str) else request.stop

        configuration = (
            cf.configuration
            if (cf := request.custom_fields) is not None
            else None
        )

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
            response_format=request.response_format,
            seed=request.seed,
            configuration=configuration,
            reasoning_effort=request.reasoning_effort,
        )

    def parse_configuration(self, cls: type[_Model]) -> _Model:
        try:
            return cls.model_validate(self.configuration or {})
        except PydanticValidationError as e:
            if self.configuration is None:
                msg = "The configuration at path 'custom_fields.configuration' is missing."
            else:
                error = e.errors()[0]
                path = ".".join(map(str, error["loc"]))
                msg = f"Invalid request. Path: 'custom_fields.configuration.{path}', error: {error['msg']}"

            raise RequestValidationError(msg) from None


def get_attachments(message: Message) -> list[Attachment]:
    if (custom_content := message.custom_content) is None:
        return []

    ret: list[Attachment] = []
    for attachment in custom_content.attachments or []:
        if (
            message.role == Role.ASSISTANT
            and attachment.reference_type is not None
        ):
            # Skipping citation attachments from the assistant
            continue
        ret.append(attachment)

    return ret


def collect_text_content(
    content: str | list[MessageContentPart] | None, delimiter: str = "\n\n"
) -> str:
    match content:
        case None:
            return ""
        case str():
            return content
        case list():
            texts: list[str] = []
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        texts.append(text)
                    case MessageContentImagePart():
                        raise ValidationError(
                            "Can't extract text from an image content part"
                        )
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Can't extract text from a refusal content part"
                        )
                    case MessageContentFilePart():
                        raise ValidationError(
                            "Can't extract text from a file content part"
                        )
                    case MessageContentAudioPart():
                        raise ValidationError(
                            "Can't extract text from an audio content part"
                        )
                    case _:
                        assert_never(part)
            return delimiter.join(texts)
        case _:
            assert_never(content)


def is_system_role(
    role: Role,
) -> TypeGuard[Literal[Role.SYSTEM, Role.DEVELOPER]]:
    return role in [Role.SYSTEM, Role.DEVELOPER]
