from logging import DEBUG
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    assert_never,
)

import vertexai.preview.generative_models as generative_models
from aidial_sdk.chat_completion import FinishReason, Message
from typing_extensions import override
from vertexai.preview.generative_models import (
    ChatSession,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
)

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.prompt.base import GeminiPrompt
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_0_pro import (
    Gemini_1_0_Pro_Prompt,
)
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_0_pro_vision import (
    Gemini_1_0_Pro_Vision_Prompt,
)
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_1_5_pro import (
    Gemini_1_5_Pro_Prompt,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    GeminiDeployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer
from aidial_adapter_vertexai.vertex_ai import get_gemini_model, init_vertex_ai

HarmCategory = generative_models.HarmCategory
HarmBlockThreshold = generative_models.HarmBlockThreshold
GenFinishReason = generative_models.FinishReason

default_safety_settings: Dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    # Currently n>1 is emulated by calling the model n times.
    # So the individual generation requests are expected to have n=1 or unset.
    if params.n is not None and params.n > 1:
        raise ValueError("n is expected to be 1 or unset")

    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=params.stop,
        top_p=params.top_p,
        candidate_count=params.n,
    )


class FinishReasonOtherError(Exception):
    def __init__(self, msg: str, retriable: bool):
        self.msg = msg
        self.retriable = retriable
        super().__init__(self.msg)


class GeminiChatCompletionAdapter(ChatCompletionAdapter[GeminiPrompt]):
    deployment: GeminiDeployment

    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model: GenerativeModel,
        deployment: GeminiDeployment,
    ):
        self.file_storage = file_storage
        self.model = model
        self.deployment = deployment

    @override
    async def parse_prompt(
        self, tools: ToolsConfig, messages: List[Message]
    ) -> GeminiPrompt | UserError:
        match self.deployment:
            case ChatCompletionDeployment.GEMINI_PRO_1:
                return Gemini_1_0_Pro_Prompt.parse(tools, messages)
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1:
                return await Gemini_1_0_Pro_Vision_Prompt.parse(
                    self.file_storage, tools, messages
                )
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1_5:
                return await Gemini_1_5_Pro_Prompt.parse(
                    self.file_storage, tools, messages
                )
            case _:
                assert_never(self.deployment)

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiPrompt
    ) -> AsyncIterator[GenerationResponse]:
        session = ChatSession(model=self.model, history=prompt.history)
        parameters = create_generation_config(params)

        if params.stream:
            response = await session._send_message_streaming_async(
                content=prompt.prompt,  # type: ignore
                generation_config=parameters,
                safety_settings=default_safety_settings,
                tools=prompt.tools,
            )

            async for chunk in response:
                yield chunk
        else:
            response = await session._send_message_async(
                content=prompt.prompt,  # type: ignore
                generation_config=parameters,
                safety_settings=default_safety_settings,
                tools=prompt.tools,
            )

            yield response

    @staticmethod
    async def process_chunks(
        consumer: Consumer,
        generator: Callable[[], AsyncIterator[GenerationResponse]],
    ) -> AsyncIterator[str]:
        no_content_generated = True

        async for chunk in generator():
            if log.isEnabledFor(DEBUG):
                chunk_str = json_dumps(chunk, excluded_keys=["safety_ratings"])
                log.debug(f"response chunk: {chunk_str}")

            await set_finish_reason(chunk, consumer, no_content_generated)
            await set_usage(chunk, consumer)

            content = get_content(chunk)
            if content is not None:
                no_content_generated = no_content_generated and content == ""
                yield content

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
                    lambda: self.send_message_async(params, prompt),
                ),
                2,
            ):
                completion += content
                await consumer.append_content(content)

            log.debug(f"predict response: {completion!r}")

    @override
    async def count_prompt_tokens(self, prompt: GeminiPrompt) -> int:
        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = await self.model.count_tokens_async(prompt.contents)
            log.debug(f"count_tokens[prompt] response: {json_dumps(resp)}")
            return resp.total_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = await self.model.count_tokens_async(string)
            log.debug(f"count_tokens[completion] response: {json_dumps(resp)}")
            return resp.total_tokens

    @classmethod
    async def create(
        cls,
        file_storage: Optional[FileStorage],
        model_id: str,
        deployment: GeminiDeployment,
        project_id: str,
        location: str,
    ) -> "GeminiChatCompletionAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_gemini_model(model_id)
        return cls(file_storage, model, deployment)


def get_content(response: GenerationResponse) -> Optional[str]:
    try:
        return response.text
    except Exception:
        return None


async def set_finish_reason(
    response: GenerationResponse, consumer: Consumer, no_content_generated: bool
) -> None:
    finish_reason = response.candidates[0].finish_reason

    openai_finish_reason = to_openai_finish_reason(
        finish_reason=finish_reason,
        retriable=no_content_generated,
    )

    if openai_finish_reason is not None:
        await consumer.set_finish_reason(openai_finish_reason)


async def set_usage(response: GenerationResponse, consumer: Consumer) -> None:
    if "usage_metadata" in response._raw_response:
        usage = response._raw_response.usage_metadata
        log.debug(f"usage: {json_dumps(usage)}")

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                completion_tokens=usage.candidates_token_count,
            )
        )


def to_openai_finish_reason(
    finish_reason: GenFinishReason, retriable: bool
) -> FinishReason | None:
    match finish_reason:
        case GenFinishReason.FINISH_REASON_UNSPECIFIED:
            return None
        case GenFinishReason.MAX_TOKENS:
            return FinishReason.LENGTH
        case GenFinishReason.STOP:
            return FinishReason.STOP
        case (
            GenFinishReason.SAFETY
            | GenFinishReason.RECITATION
            | GenFinishReason.BLOCKLIST
            | GenFinishReason.PROHIBITED_CONTENT
            | GenFinishReason.SPII
        ):
            return FinishReason.CONTENT_FILTER
        case GenFinishReason.OTHER:
            # OTHER finish reason could be usually fixed with a retry
            raise FinishReasonOtherError(
                msg="The model terminated generation unexpectedly",
                retriable=retriable,
            )
        case _:
            assert_never(finish_reason)


T = TypeVar("T")


async def generate_with_retries(
    generator: Callable[[], AsyncIterator[T]], max_retries: int
) -> AsyncIterator[T]:
    retries = 0
    while True:
        try:
            async for content in generator():
                yield content
            break

        except FinishReasonOtherError as e:
            if not e.retriable:
                raise e

            retries += 1
            if retries > max_retries:
                log.debug(f"max retries exceeded ({max_retries})")
                raise e

            log.debug(f"retrying [{retries}/{max_retries}]")
