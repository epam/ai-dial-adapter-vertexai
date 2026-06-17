from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from logging import DEBUG
from typing import Any

from aidial_sdk.chat_completion import FinishReason, Message
from aidial_sdk.exceptions import RuntimeServerError
from google.genai.client import Client as GenAIClient
from google.genai.types import CountTokensConfigDict as GenAICountTokensConfig
from google.genai.types import (
    GenerateContentConfigDict as GenAIGenerationConfig,
)
from google.genai.types import (
    GenerateContentResponse as GenAIGenerateContentResponse,
)
from google.genai.types import ImageConfigDict, ThinkingConfigDict
from pydantic import Field
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.error import generate_with_retries
from aidial_adapter_vertexai.chat.gemini.finish_reason import (
    genai_to_openai_finish_reason,
    invalid_tool_call_message,
)
from aidial_adapter_vertexai.chat.gemini.generation_config import (
    create_genai_count_tokens_config,
    create_genai_generation_config,
)
from aidial_adapter_vertexai.chat.gemini.grounding import create_grounding
from aidial_adapter_vertexai.chat.gemini.input import to_genai_thinking_level
from aidial_adapter_vertexai.chat.gemini.output import (
    create_citations,
    create_function_calls_from_genai,
    create_image_attachment,
    set_usage,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPromptGenAI
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_2 import Gemini_2_Prompt
from aidial_adapter_vertexai.chat.gemini.stage import (
    CodeExecutionStage,
    LazyStage,
)
from aidial_adapter_vertexai.chat.gemini.state import MessageState
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.deployments import GeminiDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.pydantic import (
    ExtraAllowModel,
    ExtraForbidModel,
)
from aidial_adapter_vertexai.utils.timer import Timer


class ImageConfig(ExtraAllowModel):
    """The image generation configuration"""

    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio of the generated images. "
            'Supported values are "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", and "21:9".'
        ),
    )

    image_size: str | None = Field(
        default=None,
        description=(
            "Specifies the size of generated images. "
            "Supported values are `1K`, `2K`, `4K`. "
            "If not specified, the model will use default value `1K`."
        ),
    )

    def to_image_config(self) -> ImageConfigDict:
        ret: ImageConfigDict = {
            "aspect_ratio": self.aspect_ratio,
            "image_size": self.image_size,
        }
        return ret | self.extra_fields  # type: ignore


class ThinkingConfig(ExtraAllowModel):
    """The thinking features configuration."""

    include_thoughts: bool | None = Field(
        default=None,
        description="Whether to include thoughts in the response. If true, thoughts are returned in a dedicated stage given that the thoughts are available.",
    )

    thinking_budget: int | None = Field(
        default=None, description="The thinking budget in tokens."
    )

    def to_thinking_config(self) -> ThinkingConfigDict:
        ret: ThinkingConfigDict = {
            "include_thoughts": self.include_thoughts,
            "thinking_budget": self.thinking_budget,
        }
        return ret | self.extra_fields  # type: ignore


class GeminiConfigurationModel(ExtraForbidModel):
    thinking: ThinkingConfig | None = None
    image_config: ImageConfig | None = None


@dataclass
class GeminiConfiguration:
    supports_thinking: bool
    supports_image_generation: bool

    def model_json_schema(self) -> dict[str, Any]:
        schema = GeminiConfigurationModel.model_json_schema()
        props = schema["properties"]

        if not self.supports_thinking:
            props.pop("thinking")

        if not self.supports_image_generation:
            props.pop("image_config")

        return schema


class GeminiGenAIChatCompletionAdapter(
    ChatCompletionAdapter[GeminiPromptGenAI]
):
    deployment: AdapterDeployment[GeminiDeployment]
    client: GenAIClient

    def __init__(
        self,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[GeminiDeployment],
        client: GenAIClient,
    ):
        self.file_storage = file_storage
        self.deployment = deployment
        self.client = client

    @property
    def supports_thinking(self) -> bool:
        deployment = self.deployment.reference_deployment_id
        if deployment in (
            D.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
            D.GEMINI_2_5_FLASH_IMAGE,
            D.GEMINI_3_1_FLASH_IMAGE_PREVIEW,
        ):
            return False
        return (
            "gemini-2.5" in deployment.value or "gemini-3" in deployment.value
        )

    @property
    def supports_image_generation(self) -> bool:
        return self.deployment.reference_deployment_id in (
            D.GEMINI_2_0_FLASH_EXP,
            D.GEMINI_2_5_FLASH_IMAGE_PREVIEW,
            D.GEMINI_2_5_FLASH_IMAGE,
            D.GEMINI_3_PRO_IMAGE_PREVIEW,
            D.GEMINI_3_1_FLASH_IMAGE_PREVIEW,
        )

    async def configuration(self) -> GeminiConfiguration | None:
        if self.supports_image_generation or self.supports_thinking:
            return GeminiConfiguration(
                supports_image_generation=self.supports_image_generation,
                supports_thinking=self.supports_thinking,
            )
        return None

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[GeminiDeployment],
        config: UpstreamConfig,
    ) -> "GeminiGenAIChatCompletionAdapter":
        client = await config.get_genai_client()
        return cls(file_storage, deployment, client)

    @property
    def model_id(self) -> str:
        return self.deployment.upstream_deployment_id

    @override
    async def parse_prompt(
        self,
        params: ModelParameters,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: list[Message],
    ) -> GeminiPromptGenAI | UserError:
        return await Gemini_2_Prompt.parse(
            self.file_storage, tools, static_tools, messages
        )

    async def _get_generation_config(
        self, params: ModelParameters, prompt: GeminiPromptGenAI
    ) -> GenAIGenerationConfig:
        configuration = params.parse_configuration(GeminiConfigurationModel)

        image_config: ImageConfigDict | None = None
        if configuration and configuration.image_config:
            image_config = configuration.image_config.to_image_config()

        thinking_config: ThinkingConfigDict | None = None
        if effort := params.reasoning_effort:
            thinking_config = thinking_config or {}
            thinking_config = thinking_config | {
                "thinking_level": to_genai_thinking_level(effort)
            }

        if configuration and configuration.thinking:
            thinking_config = thinking_config or {}
            thinking_config = (
                thinking_config | configuration.thinking.to_thinking_config()
            )

        return create_genai_generation_config(
            params,
            supports_image_generation=self.supports_image_generation,
            tools=prompt.tools,
            static_tools=prompt.static_tools,
            system_instruction=prompt.system,
            thinking_config=thinking_config,
            image_config=image_config,
        )

    def _get_token_count_config(
        self, prompt: GeminiPromptGenAI
    ) -> GenAICountTokensConfig:
        return create_genai_count_tokens_config(
            prompt.tools,
            prompt.static_tools,
            prompt.system,
        )

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiPromptGenAI
    ) -> AsyncIterator[GenAIGenerateContentResponse]:
        generation_config = await self._get_generation_config(params, prompt)
        contents = prompt.messages.raw_list

        if log.isEnabledFor(DEBUG):
            generation_str = json_dumps_short(
                {"config": generation_config, "contents": contents},
                exclude_none=True,
            )
            log.debug(f"generation: {generation_str}")

        if params.stream:
            gen = await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=list(contents),
                config=generation_config,
            )
            async for chunk in gen:  # type: ignore
                yield chunk
        else:
            yield await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=list(contents),
                config=generation_config,
            )

    async def process_chunks(
        self,
        consumer: Consumer,
        tools: ToolsConfig,
        generator: Callable[[], AsyncIterator[GenAIGenerateContentResponse]],
    ):
        thinking_stage = LazyStage(consumer, "Thinking")
        code_execution_stage = CodeExecutionStage(consumer, "Code execution")
        state = MessageState()

        usage_metadata = None
        is_grounding_added = False

        try:
            async for chunk in generator():
                if log.isEnabledFor(DEBUG):
                    chunk_str = json_dumps(chunk, exclude_none=True)
                    log.debug(f"response chunk: {chunk_str}")

                if (feedback := chunk.prompt_feedback) and (
                    feedback.block_reason or feedback.block_reason_message
                ):
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
                        if part.thought_signature:
                            state.set_thought_signature(part.thought_signature)

                        await create_function_calls_from_genai(
                            part, consumer, tools
                        )

                        if text := part.text:
                            if part.thought:
                                await thinking_stage.append_content(text)
                            else:
                                await consumer.append_content(text)

                            yield text
                        if (code := part.executable_code) and code.code:
                            await code_execution_stage.append_code(
                                code.code, code.language
                            )
                            yield code.code
                        if (res := part.code_execution_result) and res.output:
                            await code_execution_stage.append_code_output(
                                res.output
                            )
                            yield res.output

                        await create_image_attachment(
                            consumer, self.file_storage, part
                        )

                is_grounding_added |= await create_grounding(
                    candidate, consumer
                )

                await create_citations(candidate, consumer)
                openai_reason = genai_to_openai_finish_reason(
                    candidate.finish_reason,
                    candidate.finish_message,
                    consumer.is_empty(),
                )
                if openai_reason:
                    recovery = invalid_tool_call_message(
                        candidate.finish_reason, candidate.finish_message
                    )
                    if recovery and consumer.is_empty():
                        await consumer.append_content(recovery)
                    await consumer.set_finish_reason(openai_reason)
        finally:
            thinking_stage.close()
            await code_execution_stage.close()

            # It's possible that max tokens will be reached during the thinking stage
            # and there will be no content in response.
            # And set_usage will fail with 'Trying to set "usage" before generating all choices' error.
            # Append empty content, so at least one choice is generated.
            if consumer.get_finish_reason() is not None:
                await consumer.append_content("")

            await consumer.set_state(state.to_json())

        if usage_metadata:
            await set_usage(
                usage_metadata,
                consumer,
                self.deployment.reference_deployment_id,
                is_grounding_added,
            )

    @override
    async def truncate_prompt(
        self, prompt: GeminiPromptGenAI, max_prompt_tokens: int
    ) -> TruncatedPrompt[GeminiPromptGenAI]:
        prompt = await prompt.truncate(
            tokenize=self.count_prompt_tokens, user_limit=max_prompt_tokens
        )

        return TruncatedPrompt(
            prompt=prompt,
            discarded_messages=list(prompt.messages.get_removed_indices()),
        )

    @override
    async def count_prompt_tokens(self, prompt: GeminiPromptGenAI) -> int:
        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            config = self._get_token_count_config(prompt)
            resp = await self.client.aio.models.count_tokens(
                model=self.model_id,
                contents=list(prompt.messages.raw_list),
                config=config,
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
        prompt: GeminiPromptGenAI,
    ) -> None:
        with Timer("predict timing: {time}", log.debug):
            if log.isEnabledFor(DEBUG):
                request_str = json_dumps_short(
                    {"parameters": params, "prompt": prompt}, exclude_none=True
                )
                log.debug(f"predict request: {request_str}")

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
