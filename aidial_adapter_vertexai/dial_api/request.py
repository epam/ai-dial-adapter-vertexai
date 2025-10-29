from typing import (
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    TypeGuard,
    TypeVar,
    assert_never,
)

from aidial_sdk.chat_completion import (
    Attachment,
    Message,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    ResponseFormat,
    Role,
)
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    MessageContentRefusalPart,
)
from aidial_sdk.exceptions import RequestValidationError
from pydantic.v1 import BaseModel
from pydantic.v1 import ValidationError as PydanticValidationError

from aidial_adapter_vertexai.chat.errors import ValidationError

_Model = TypeVar("_Model", bound=BaseModel)


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
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    configuration: Optional[dict] = None

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
        )

    def parse_configuration(self, cls: Type[_Model]) -> _Model:
        try:
            return cls.parse_obj(self.configuration or {})
        except PydanticValidationError as e:
            if self.configuration is None:
                msg = "The configuration at path 'custom_fields.configuration' is missing."
            else:
                error = e.errors()[0]
                path = ".".join(map(str, error["loc"]))
                msg = f"Invalid request. Path: 'custom_fields.configuration.{path}', error: {error['msg']}"

            raise RequestValidationError(msg)


def get_attachments(message: Message) -> List[Attachment]:
    if (custom_content := message.custom_content) is None:
        return []

    ret: List[Attachment] = []
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
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Can't extract text from a refusal content part"
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
