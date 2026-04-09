import asyncio
from typing import List, assert_never

from aidial_sdk.chat_completion import (
    ChatCompletion,
    ConfigurationRequest,
    Request,
    Response,
)
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
from aidial_sdk.exceptions import ResourceNotFoundError
from typing_extensions import override

from aidial_adapter_vertexai.adapters import get_chat_completion_model
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
    TruncatedPrompt,
)
from aidial_adapter_vertexai.chat.consumer import ChoiceConsumer
from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment
from aidial_adapter_vertexai.dial_api.caching import (
    set_response_headers_for_caching,
)
from aidial_adapter_vertexai.dial_api.exceptions import (
    dial_exception_decorator,
    to_dial_exception,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.upstream_config import parse_upstream_config
from aidial_adapter_vertexai.utils.adapter_deployments import (
    AdapterChatCompletionDeployment,
    resolve_upstream_deployment_id_from_request,
)
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.not_implemented import is_implemented


class VertexAIChatCompletion(ChatCompletion):
    def _get_deployment(
        self, request: FromRequestDeploymentMixin
    ) -> AdapterChatCompletionDeployment:
        return resolve_upstream_deployment_id_from_request(
            ChatCompletionDeployment, request
        )

    async def _get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        return await get_chat_completion_model(
            api_key=request.api_key,
            deployment=self._get_deployment(request),
            upstream_config=parse_upstream_config(request),
        )

    @override
    @dial_exception_decorator
    async def configuration(self, request: ConfigurationRequest):
        model = await self._get_model(request)
        if (
            not is_implemented(model.configuration)
            or (cls := await model.configuration()) is None
        ):
            raise ResourceNotFoundError("The endpoint is not implemented")
        return cls.model_json_schema()

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        deployment = self._get_deployment(request)
        response.set_model(deployment.upstream_deployment_id)

        params = ModelParameters.create(request)
        # Currently n>1 is emulated by calling the model n times
        n = params.n or 1
        params.n = None

        max_prompt_tokens = params.max_prompt_tokens
        params.max_prompt_tokens = None

        model = await self._get_model(request)
        tools = ToolsConfig.from_request(request)
        static_tools = StaticToolsConfig.from_request(request)
        prompt = await model.parse_prompt(
            params, tools, static_tools, request.messages
        )

        if isinstance(prompt, UserError):
            await prompt.report_usage(response)
            raise prompt

        if max_prompt_tokens is None:
            truncated_prompt = TruncatedPrompt(
                prompt=prompt, discarded_messages=[]
            )
        else:
            if not is_implemented(model.truncate_prompt):
                raise ValidationError(
                    "max_prompt_tokens request parameter is not supported"
                )
            truncated_prompt = await model.truncate_prompt(
                prompt, max_prompt_tokens
            )

        async def set_response_headers(prompt_tokens: int | None = None):
            async def _get_request_tokens() -> int:
                return prompt_tokens or await model.count_prompt_tokens(
                    truncated_prompt.prompt
                )

            deployment = self._get_deployment(request)
            await set_response_headers_for_caching(
                response,
                deployment=deployment.reference_deployment_id,
                request_headers=request.original_request.headers,
                request_body=await request.original_request.json(),
                get_request_tokens=_get_request_tokens,
            )

        async def generate_response(usage: TokenUsage) -> None:
            with ChoiceConsumer(response=response) as consumer:
                await model.chat(params, consumer, truncated_prompt.prompt)
                usage.accumulate(consumer.get_usage())

            log.debug(
                f"finish_reason[{consumer.choice_idx}]: {consumer.finish_reason}"
            )

        if request.stream:
            await set_response_headers()

        usage = TokenUsage()

        await asyncio.gather(*(generate_response(usage) for _ in range(n)))

        log.debug(f"usage: {usage}")
        usage.set_response_usage(response)

        if not request.stream:
            await set_response_headers(usage.prompt_tokens)

        if max_prompt_tokens is not None:
            response.set_discarded_messages(truncated_prompt.discarded_messages)

    @override
    @dial_exception_decorator
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self._get_model(request)

        if not is_implemented(
            model.count_completion_tokens
        ) or not is_implemented(model.count_prompt_tokens):
            raise ResourceNotFoundError("The endpoint is not implemented")

        outputs: List[TokenizeOutput] = []
        for input in request.inputs:
            match input:
                case TokenizeInputRequest():
                    outputs.append(
                        await self._tokenize_request(model, input.value)
                    )
                case TokenizeInputString():
                    outputs.append(
                        await self._tokenize_string(model, input.value)
                    )
                case _:
                    assert_never(input.type)
        return TokenizeResponse(outputs=outputs)

    async def _tokenize_string(
        self, model: ChatCompletionAdapter, value: str
    ) -> TokenizeOutput:
        try:
            tokens = await model.count_completion_tokens(value)
            return TokenizeSuccess(token_count=tokens)
        except Exception as e:
            log.exception("Error tokenizing a string")
            return TokenizeError(error=to_dial_exception(e).message)

    async def _tokenize_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TokenizeOutput:
        try:
            params = ModelParameters.create(request)
            tools = ToolsConfig.from_request(request)
            static_tools = StaticToolsConfig.from_request(request)
            prompt = await model.parse_prompt(
                params, tools, static_tools, request.messages
            )
            if isinstance(prompt, UserError):
                raise prompt

            token_count = await model.count_prompt_tokens(prompt)
            return TokenizeSuccess(token_count=token_count)
        except Exception as e:
            log.exception("Error tokenizing a request")
            return TokenizeError(error=to_dial_exception(e).message)

    @override
    @dial_exception_decorator
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self._get_model(request)

        if not is_implemented(model.truncate_prompt):
            raise ResourceNotFoundError("The endpoint is not implemented")

        outputs: List[TruncatePromptResult] = []
        for input in request.inputs:
            outputs.append(await self._truncate_prompt_request(model, input))
        return TruncatePromptResponse(outputs=outputs)

    async def _truncate_prompt_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TruncatePromptResult:
        try:
            if request.max_prompt_tokens is None:
                raise ValidationError("max_prompt_tokens is required")

            params = ModelParameters.create(request)
            tools = ToolsConfig.from_request(request)
            static_tools = StaticToolsConfig.from_request(request)
            prompt = await model.parse_prompt(
                params, tools, static_tools, request.messages
            )

            truncated_prompt = await model.truncate_prompt(
                prompt, request.max_prompt_tokens
            )
            return TruncatePromptSuccess(
                discarded_messages=truncated_prompt.discarded_messages
            )
        except Exception as e:
            return TruncatePromptError(error=str(e))
