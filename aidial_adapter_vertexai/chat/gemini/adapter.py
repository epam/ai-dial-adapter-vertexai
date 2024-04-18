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

from aidial_sdk.chat_completion import FinishReason, Message
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
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
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    GeminiDeployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps_short, to_dict
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer
from aidial_adapter_vertexai.vertex_ai import get_gemini_model, init_vertex_ai

HarmCategory = gapic_content_types.HarmCategory
HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold
Candidate = gapic_content_types.Candidate

BLOCK_NONE_SAFETY_SETTINGS: Dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
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
        self, messages: List[Message]
    ) -> GeminiPrompt | UserError:
        match self.deployment:
            case ChatCompletionDeployment.GEMINI_PRO_1:
                return Gemini_1_0_Pro_Prompt.parse(messages)
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1:
                return await Gemini_1_0_Pro_Vision_Prompt.parse(
                    self.file_storage, messages
                )
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1_5:
                return await Gemini_1_5_Pro_Prompt.parse(
                    self.file_storage, messages
                )
            case _:
                assert_never(self.deployment)

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiPrompt
    ) -> AsyncIterator[GenerationResponse]:
        session = ChatSession(
            model=self.model, history=prompt.history, raise_on_blocked=False
        )
        parameters = create_generation_config(params)

        if params.stream:
            response = await session._send_message_streaming_async(
                content=prompt.prompt,  # type: ignore
                generation_config=parameters,
                safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                tools=None,
            )

            async for chunk in response:
                yield chunk
        else:
            response = await session._send_message_async(
                content=prompt.prompt,  # type: ignore
                generation_config=parameters,
                safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                tools=None,
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
                log.debug(f"response chunk: {to_dict(chunk)}")

            finish_reason = chunk.candidates[0].finish_reason

            openai_finish_reason = to_openai_finish_reason(
                finish_reason=finish_reason,
                retriable=no_content_generated,
            )

            if openai_finish_reason is not None:
                await consumer.set_finish_reason(openai_finish_reason)

            content = get_content(chunk)
            if content is not None:
                no_content_generated = no_content_generated and content == ""
                yield content

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: GeminiPrompt
    ) -> None:
        prompt_tokens = await self.count_prompt_tokens(prompt)

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

        completion_tokens = await self.count_completion_tokens(completion)

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @override
    async def count_prompt_tokens(self, prompt: GeminiPrompt) -> int:
        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = await self.model.count_tokens_async(prompt.contents)
            return resp.total_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = await self.model.count_tokens_async(string)
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


def to_openai_finish_reason(
    finish_reason: Candidate.FinishReason, retriable: bool
) -> FinishReason | None:
    match finish_reason:
        case Candidate.FinishReason.FINISH_REASON_UNSPECIFIED:
            return None
        case Candidate.FinishReason.MAX_TOKENS:
            return FinishReason.LENGTH
        case Candidate.FinishReason.STOP:
            return FinishReason.STOP
        case (
            Candidate.FinishReason.SAFETY
            | Candidate.FinishReason.RECITATION
            | Candidate.FinishReason.BLOCKLIST
            | Candidate.FinishReason.PROHIBITED_CONTENT
            | Candidate.FinishReason.SPII
        ):
            return FinishReason.CONTENT_FILTER
        case Candidate.FinishReason.OTHER:
            # OTHER finish reason could be usually fixed with a retry
            raise FinishReasonOtherError(
                msg="The model terminated generation unexpectedly",
                retriable=retriable,
            )
        case _:
            assert_never(finish_reason)


T = TypeVar("T")


async def generate_with_retries(
    generator: Callable[[], AsyncIterator[T]],
    max_retries: int,
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
