import asyncio
from typing import List

from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.server.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage


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
            model_params=ModelParameters.create(request),
        )

        streaming = bool(request.stream)

        async def generate_response(idx: int) -> TokenUsage:
            content, usage = await model.chat(streaming, request.messages)

            with response.create_choice() as choice:
                choice.append_content(content)
                return usage

        usages: List[TokenUsage] = await asyncio.gather(
            *(generate_response(idx) for idx in range(request.n or 1))
        )

        usage = sum(usages, TokenUsage())
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)
