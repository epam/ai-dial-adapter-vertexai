import asyncio
from typing import List, Optional, Tuple

from aidial_sdk.chat_completion import (
    ChatCompletion,
    Message,
    Request,
    Response,
    Role,
)

from aidial_adapter_vertexai.llm.consumer import ChoiceConsumer
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.llm.history_trimming import (
    get_discarded_messages_count,
)
from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_chat import (
    VertexAIAuthor,
    VertexAIMessage,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.server.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import app_logger as log

_SUPPORTED_ROLES = {Role.SYSTEM, Role.USER, Role.ASSISTANT}


def _parse_message(message: Message) -> VertexAIMessage:
    author = (
        VertexAIAuthor.BOT
        if message.role == Role.ASSISTANT
        else VertexAIAuthor.USER
    )
    return VertexAIMessage(author=author, content=message.content)  # type: ignore


def _validate_messages_and_split(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message]]:
    if len(messages) == 0:
        raise ValidationError("The chat history must have at least one message")

    for message in messages:
        if message.content is None:
            raise ValidationError("Message content must be present")

        if message.role not in _SUPPORTED_ROLES:
            raise ValidationError(
                f"Message role must be one of {_SUPPORTED_ROLES}"
            )

    context: Optional[str] = None
    if len(messages) > 0 and messages[0].role == Role.SYSTEM:
        context = messages[0].content or ""
        context = context if context.strip() else None
        messages = messages[1:]

    if len(messages) == 0 and context is not None:
        raise ValidationError(
            "The chat history must have at least one non-system message"
        )

    role: Optional[Role] = None
    for message in messages:
        if message.role == Role.SYSTEM:
            raise ValidationError(
                "System messages other than the initial system message are not allowed"
            )

        # Bison doesn't support empty messages,
        # so we replace it with a single space.
        message.content = message.content or " "

        if role == message.role:
            raise ValidationError("Messages must alternate between authors")

        role = message.role

    if len(messages) % 2 == 0:
        raise ValidationError(
            "There should be odd number of messages for correct alternating turn"
        )

    if messages[-1].role != Role.USER:
        raise ValidationError("The last message must be a user message")

    return context, messages


def _parse_history(
    history: List[Message],
) -> Tuple[Optional[str], List[VertexAIMessage]]:
    context, history = _validate_messages_and_split(history)

    return context, list(map(_parse_message, history))


class VertexAIChatCompletion(ChatCompletion):
    region: str
    project_id: str

    def __init__(self, region: str, project_id: str):
        self.region = region
        self.project_id = project_id

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        model = await get_chat_completion_model(
            deployment=ChatCompletionDeployment(request.deployment_id),
            project_id=self.project_id,
            location=self.region,
        )

        params = ModelParameters.create(request)
        context, messages = _parse_history(request.messages)
        discarded_messages_count: Optional[int] = None
        if params.max_prompt_tokens is not None:
            discarded_messages_count = await get_discarded_messages_count(
                model, context, messages, params.max_prompt_tokens
            )
            messages = messages[discarded_messages_count:]

        async def generate_response(usage: TokenUsage, choice_idx: int) -> None:
            with response.create_choice() as choice:
                consumer = ChoiceConsumer(choice)
                await model.chat(consumer, context, messages, params)
                usage.accumulate(consumer.usage)

        usage = TokenUsage()

        await asyncio.gather(
            *(generate_response(usage, idx) for idx in range(request.n or 1))
        )

        log.debug(f"usage: {usage}")
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)

        if discarded_messages_count is not None:
            response.set_discarded_messages(discarded_messages_count)
