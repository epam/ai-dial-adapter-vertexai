from logging import DEBUG
from typing import AsyncIterator, Callable, List, Optional, assert_never, cast

from aidial_sdk.chat_completion import FinishReason, Message
from typing_extensions import override
from vertexai.preview.generative_models import (
    Candidate,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
)

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.finish_reason import (
    to_openai_finish_reason,
)
from aidial_adapter_vertexai.chat.gemini.generation_config import (
    create_generation_config,
)
from aidial_adapter_vertexai.chat.gemini.grounding import create_grounding
from aidial_adapter_vertexai.chat.gemini.output import (
    create_attachments_from_citations,
    create_function_calls,
    set_usage,
)
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_0_pro import (
    Gemini_1_0_Pro_Prompt,
)
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_0_pro_vision import (
    Gemini_1_0_Pro_Vision_Prompt,
)
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5 import (
    Gemini_1_5_Prompt,
)
from aidial_adapter_vertexai.chat.gemini.utils import generate_with_retries
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    GeminiDeployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer


def _get_candidate_text_safe(candidate: Candidate) -> str | None:
    # The text content of a candidate may be missing when function is called or
    # when the generation was terminated with SAFETY finish reason.
    try:
        return candidate.text
    except ValueError as e:
        log.debug(f"The Candidate doesn't have text: {e}")
        return None


class GeminiChatCompletionAdapter(ChatCompletionAdapter[GeminiPrompt]):
    deployment: GeminiDeployment

    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model_id: str,
        deployment: GeminiDeployment,
    ):
        self.file_storage = file_storage
        self.model_id = model_id
        self.deployment = deployment

    @override
    async def parse_prompt(
        self,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> GeminiPrompt | UserError:
        match self.deployment:
            case ChatCompletionDeployment.GEMINI_PRO_1:
                return await Gemini_1_0_Pro_Prompt.parse(
                    tools, static_tools, messages
                )
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1:
                return await Gemini_1_0_Pro_Vision_Prompt.parse(
                    self.file_storage, tools, static_tools, messages
                )
            case (
                ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW
                | ChatCompletionDeployment.GEMINI_PRO_1_5_V1
                | ChatCompletionDeployment.GEMINI_PRO_1_5_V2
                | ChatCompletionDeployment.GEMINI_FLASH_1_5_V1
                | ChatCompletionDeployment.GEMINI_FLASH_1_5_V2
            ):
                return await Gemini_1_5_Prompt.parse(
                    self.file_storage, tools, static_tools, messages
                )
            case _:
                assert_never(self.deployment)

    def _get_model(
        self,
        *,
        params: ModelParameters | None = None,
        prompt: GeminiPrompt | None = None,
    ) -> GenerativeModel:
        parameters = create_generation_config(params) if params else None

        if prompt is not None:
            tools = prompt.to_gemini_tools() or None
            tool_config = prompt.tools.to_gemini_tool_config()
            system_instruction = cast(
                List[str | Part | Image] | None,
                prompt.system_instruction,
            )
        else:
            tools = None
            tool_config = None
            system_instruction = None

        return GenerativeModel(
            self.model_id,
            generation_config=parameters,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
        )

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiPrompt
    ) -> AsyncIterator[GenerationResponse]:

        model = self._get_model(params=params, prompt=prompt)
        contents = prompt.contents

        if params.stream:
            response = await model._generate_content_streaming_async(contents)

            async for chunk in response:
                yield chunk
        else:
            yield await model._generate_content_async(contents)

    async def process_chunks(
        self,
        consumer: Consumer,
        tools: ToolsConfig,
        generator: Callable[[], AsyncIterator[GenerationResponse]],
    ) -> AsyncIterator[str]:

        usage_metadata = None
        is_grounding_added = False

        async for chunk in generator():
            if log.isEnabledFor(DEBUG):
                chunk_str = json_dumps(chunk, excluded_keys=["safety_ratings"])
                log.debug(f"response chunk: {chunk_str}")
            if chunk.candidates:
                candidate = chunk.candidates[0]

                if (content := _get_candidate_text_safe(candidate)) is not None:
                    await consumer.append_content(content)
                    yield content

                await create_function_calls(candidate, consumer, tools)
                is_grounding_added |= await create_grounding(
                    candidate, consumer
                )

                await create_attachments_from_citations(candidate, consumer)
                if openai_reason := to_openai_finish_reason(
                    candidate.finish_reason,
                    consumer.is_empty(),
                ):
                    await consumer.set_finish_reason(openai_reason)

            if chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
            if chunk.prompt_feedback:
                await consumer.set_finish_reason(FinishReason.CONTENT_FILTER)

        if usage_metadata:
            await set_usage(
                usage_metadata,
                consumer,
                self.deployment,
                is_grounding_added,
            )

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: GeminiPrompt
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

    @override
    async def truncate_prompt(
        self, prompt: GeminiPrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[GeminiPrompt]:
        return await prompt.truncate(
            tokenizer=self.count_prompt_tokens, user_limit=max_prompt_tokens
        )

    @override
    async def count_prompt_tokens(self, prompt: GeminiPrompt) -> int:
        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = await self._get_model(prompt=prompt).count_tokens_async(
                prompt.contents
            )
            log.debug(f"count_tokens[prompt] response: {json_dumps(resp)}")
            return resp.total_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = await self._get_model().count_tokens_async(string)
            log.debug(f"count_tokens[completion] response: {json_dumps(resp)}")
            return resp.total_tokens

    @classmethod
    async def create(
        cls,
        file_storage: Optional[FileStorage],
        model_id: str,
        deployment: GeminiDeployment,
    ) -> "GeminiChatCompletionAdapter":
        return cls(file_storage, model_id, deployment)
