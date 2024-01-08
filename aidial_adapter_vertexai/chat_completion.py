import asyncio

from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import ChoiceConsumer
from aidial_adapter_vertexai.llm.exceptions import UserError
from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.server.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.universal_api.errors import report_user_error
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
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

        params = ModelParameters.create(request)
        prompt = await model.parse_prompt(request.messages)

        discarded_messages_count = 0
        if params.max_prompt_tokens is not None and not isinstance(
            prompt, UserError
        ):
            prompt, discarded_messages_count = await model.truncate_prompt(
                prompt, params.max_prompt_tokens
            )

        async def generate_response(usage: TokenUsage, choice_idx: int) -> None:
            with response.create_choice() as choice:
                consumer = ChoiceConsumer(choice)
                if isinstance(prompt, UserError):
                    await report_user_error(choice, headers, prompt)
                else:
                    await model.chat(params, consumer, prompt)
                    usage.accumulate(consumer.usage)

        usage = TokenUsage()

        await asyncio.gather(
            *(generate_response(usage, idx) for idx in range(request.n or 1))
        )

        log.debug(f"usage: {usage}")
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)

        if params.max_prompt_tokens is not None:
            response.set_discarded_messages(discarded_messages_count)
