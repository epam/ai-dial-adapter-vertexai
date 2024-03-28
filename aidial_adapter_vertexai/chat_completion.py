import asyncio
from typing import List, assert_never

from aidial_sdk.chat_completion import ChatCompletion, Request, Response, Status
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.deployment.tokenize import (
    TokenizeError,
    TokenizeInputRequest,
    TokenizeInputString,
    TokenizeOutput,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeSuccess,
)
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptResult,
    TruncatePromptSuccess,
)
from typing_extensions import override

from aidial_adapter_vertexai.adapters import get_chat_completion_model
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import ChoiceConsumer
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import app_logger as log


class VertexAIChatCompletion(ChatCompletion):
    region: str
    project_id: str

    def __init__(self, region: str, project_id: str):
        self.region = region
        self.project_id = project_id

    async def get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        return await get_chat_completion_model(
            deployment=ChatCompletionDeployment(request.deployment_id),
            project_id=self.project_id,
            location=self.region,
            headers=request.headers,
        )

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        model = await self.get_model(request)
        prompt = await model.parse_prompt(request.messages)

        if isinstance(prompt, UserError):
            # Show the error message in a stage for a web UI user
            with response.create_choice() as choice:
                stage = choice.create_stage("Error")
                stage.open()
                stage.append_content(prompt.to_message_for_chat_user())
                stage.close(Status.FAILED)
            await response.aflush()

            # Raise exception for a DIAL API client
            raise Exception(prompt.message)

        params = ModelParameters.create(request)

        # Currently n>1 is emulated by calling the model n times
        n = params.n or 1
        params.n = None

        if n > 1 and params.stream:
            raise ValidationError("n>1 is not supported in streaming mode")

        discarded_messages: List[int] = []
        if params.max_prompt_tokens is not None:
            prompt, discarded_messages = await model.truncate_prompt(
                prompt, params.max_prompt_tokens
            )

        async def generate_response(usage: TokenUsage, choice_idx: int) -> None:
            choice = response.create_choice()
            choice.open()

            consumer = ChoiceConsumer(choice)
            await model.chat(params, consumer, prompt)
            usage.accumulate(consumer.usage)

            finish_reason = consumer.finish_reason
            log.debug(f"finish_reason[{choice_idx}]: {finish_reason}")
            choice.close(finish_reason)

        usage = TokenUsage()

        await asyncio.gather(
            *(generate_response(usage, idx) for idx in range(n))
        )

        log.debug(f"usage: {usage}")
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)

        if params.max_prompt_tokens is not None:
            response.set_discarded_messages(discarded_messages)

    @override
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self.get_model(request)

        outputs: List[TokenizeOutput] = []
        for input in request.inputs:
            match input:
                case TokenizeInputRequest():
                    outputs.append(
                        await self.tokenize_request(model, input.value)
                    )
                case TokenizeInputString():
                    outputs.append(
                        await self.tokenize_string(model, input.value)
                    )
                case _:
                    assert_never(input.type)
        return TokenizeResponse(outputs=outputs)

    async def tokenize_string(
        self, model: ChatCompletionAdapter, value: str
    ) -> TokenizeOutput:
        try:
            tokens = await model.count_completion_tokens(value)
            return TokenizeSuccess(token_count=tokens)
        except Exception as e:
            return TokenizeError(error=str(e))

    async def tokenize_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TokenizeOutput:
        try:
            prompt = await model.parse_prompt(request.messages)
            if isinstance(prompt, UserError):
                raise prompt

            token_count = await model.count_prompt_tokens(prompt)
            return TokenizeSuccess(token_count=token_count)
        except Exception as e:
            return TokenizeError(error=str(e))

    @override
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self.get_model(request)
        outputs: List[TruncatePromptResult] = []
        for input in request.inputs:
            outputs.append(await self.truncate_prompt_request(model, input))
        return TruncatePromptResponse(outputs=outputs)

    async def truncate_prompt_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TruncatePromptResult:
        try:
            if request.max_prompt_tokens is None:
                raise ValueError("max_prompt_tokens is required")

            _prompt, discarded_messages = await model.truncate_prompt(
                request.messages, request.max_prompt_tokens
            )
            return TruncatePromptSuccess(discarded_messages=discarded_messages)
        except Exception as e:
            return TruncatePromptError(error=str(e))
