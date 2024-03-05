import asyncio
from typing import List

from aidial_sdk.chat_completion import ChatCompletion, Request, Response, Status

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

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        headers = request.headers
        model: ChatCompletionAdapter = await get_chat_completion_model(
            deployment=ChatCompletionDeployment(request.deployment_id),
            project_id=self.project_id,
            location=self.region,
            headers=headers,
        )

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
