from logging import DEBUG
from typing import AsyncIterator, Callable, List, Optional, assert_never

from aidial_sdk.chat_completion import FinishReason, Message, Stage
from aidial_sdk.exceptions import RuntimeServerError
from google.genai.client import Client as GenAIClient
from google.genai.types import (
    GenerateContentResponse as GenAIGenerateContentResponse,
)
from typing_extensions import override

from aidial_adapter_vertexai.app_config import get_genai_client
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.error import generate_with_retries
from aidial_adapter_vertexai.chat.gemini.finish_reason import (
    genai_to_openai_finish_reason,
)
from aidial_adapter_vertexai.chat.gemini.generation_config import (
    create_genai_generation_config,
)
from aidial_adapter_vertexai.chat.gemini.grounding import create_grounding
from aidial_adapter_vertexai.chat.gemini.output import (
    create_attachments_from_citations,
    create_function_calls_from_genai,
    set_usage,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiGenAIPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_2 import Gemini_2_Prompt
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    Gemini2Deployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer


class GeminiGenAIChatCompletionAdapter(
    ChatCompletionAdapter[GeminiGenAIPrompt]
):
    deployment: Gemini2Deployment
    client: GenAIClient

    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model_id: str,
        deployment: Gemini2Deployment,
        client: GenAIClient,
    ):
        self.file_storage = file_storage
        self.model_id = model_id
        self.deployment = deployment
        self.client = client

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        model_id: str,
        deployment: Gemini2Deployment,
        location: str,
    ) -> "GeminiGenAIChatCompletionAdapter":
        return cls(
            file_storage, model_id, deployment, get_genai_client(location)
        )

    @override
    async def parse_prompt(
        self,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> GeminiGenAIPrompt | UserError:
        match self.deployment:
            case (
                ChatCompletionDeployment.GEMINI_2_0_EXPERIMENTAL_1206
                | ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP
                | ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_1219
            ):
                return await Gemini_2_Prompt.parse(
                    self.file_storage, tools, static_tools, messages
                )
            case _:
                assert_never(self.deployment)

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiGenAIPrompt
    ) -> AsyncIterator[GenAIGenerateContentResponse]:

        generation_config = create_genai_generation_config(
            params,
            prompt.tools,
            prompt.static_tools,
            prompt.system_instruction,
        )

        if params.stream:
            async for chunk in self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=list(prompt.contents),
                config=generation_config,
            ):
                yield chunk
        else:
            yield await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=list(prompt.contents),
                config=generation_config,
            )

    async def process_chunks(
        self,
        consumer: Consumer,
        tools: ToolsConfig,
        generator: Callable[[], AsyncIterator[GenAIGenerateContentResponse]],
    ):
        thinking_stage: Stage | None = None

        usage_metadata = None
        is_grounding_added = False
        try:
            async for chunk in generator():
                if log.isEnabledFor(DEBUG):
                    chunk_str = json_dumps(chunk)
                    log.debug(f"response chunk: {chunk_str}")

                if chunk.prompt_feedback:
                    await consumer.set_finish_reason(
                        FinishReason.CONTENT_FILTER
                    )

                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata

                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        await create_function_calls_from_genai(
                            part, consumer, tools
                        )
                        if part.thought and part.text:
                            if thinking_stage is None:
                                thinking_stage = await consumer.create_stage(
                                    "Thought Process"
                                )
                                thinking_stage.open()
                            thinking_stage.append_content(part.text)
                            yield part.text
                        elif part.text:
                            await consumer.append_content(part.text)
                            yield part.text

                is_grounding_added |= await create_grounding(
                    candidate, consumer
                )

                await create_attachments_from_citations(candidate, consumer)
                if openai_reason := genai_to_openai_finish_reason(
                    candidate.finish_reason,
                    consumer.is_empty(),
                ):
                    await consumer.set_finish_reason(openai_reason)
        finally:
            if thinking_stage:
                thinking_stage.close()
            # It's possible that max tokens will be reached during the thinking stage
            # and there will be no content in response.
            # And set_usage will fail with 'Trying to set "usage" before generating all choices' error.
            # Append empty content, so at least one choice is generated.
            await consumer.append_content("")

        if usage_metadata:
            await set_usage(
                usage_metadata,
                consumer,
                self.deployment,
                is_grounding_added,
            )

    @override
    async def truncate_prompt(
        self, prompt: GeminiGenAIPrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[GeminiGenAIPrompt]:
        return await prompt.truncate(
            tokenizer=self.count_prompt_tokens, user_limit=max_prompt_tokens
        )

    @override
    async def count_prompt_tokens(self, prompt: GeminiGenAIPrompt) -> int:
        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = await self.client.aio.models.count_tokens(
                model=self.model_id,
                contents=list(prompt.contents),
            )
            log.debug(f"count_tokens[prompt] response: {json_dumps(resp)}")
            if resp.total_tokens is None:
                raise RuntimeServerError("Failed to count tokens for prompt")
            return resp.total_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = await self.client.aio.models.count_tokens(
                model=self.model_id,
                contents=string,
            )
            log.debug(f"count_tokens[completion] response: {json_dumps(resp)}")
            if resp.total_tokens is None:
                raise RuntimeServerError(
                    "Failed to count tokens for completion"
                )
            return resp.total_tokens

    @override
    async def chat(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: GeminiGenAIPrompt,
    ) -> None:

        with Timer("predict timing: {time}", log.debug):
            if log.isEnabledFor(DEBUG):
                log.debug(
                    "predict request: "
                    + json_dumps_short({"parameters": params, "prompt": prompt})
                )

            completion = ""
            async for content in generate_with_retries(
                lambda: self.process_chunks(
                    consumer,
                    prompt.tools,
                    lambda: self.send_message_async(params, prompt),
                ),
                2,
            ):
                completion += content

            log.debug(f"predict response: {completion!r}")
