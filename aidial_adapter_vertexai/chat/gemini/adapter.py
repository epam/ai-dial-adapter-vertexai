import json
from logging import DEBUG
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    assert_never,
    cast,
)

import vertexai.preview.generative_models as generative_models
from aidial_sdk.chat_completion import Attachment, FinishReason, Message
from google.cloud.aiplatform_v1beta1.types.prediction_service import (
    GenerateContentResponse,
)
from typing_extensions import override
from vertexai.preview.generative_models import (
    Candidate,
    GenerationConfig,
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
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import DiscardedMessages
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    GeminiDeployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps, json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.protobuf import recurse_proto_marshal_to_dict
from aidial_adapter_vertexai.utils.timer import Timer

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
        super().__init__(self.msg)
        self.msg = msg
        self.retriable = retriable


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
        self, tools: ToolsConfig, messages: List[Message]
    ) -> GeminiPrompt | UserError:
        match self.deployment:
            case ChatCompletionDeployment.GEMINI_PRO_1:
                return await Gemini_1_0_Pro_Prompt.parse(tools, messages)
            case ChatCompletionDeployment.GEMINI_PRO_VISION_1:
                return await Gemini_1_0_Pro_Vision_Prompt.parse(
                    self.file_storage, tools, messages
                )
            case (
                ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW
                | ChatCompletionDeployment.GEMINI_PRO_1_5_V1
                | ChatCompletionDeployment.GEMINI_PRO_1_5_V2
                | ChatCompletionDeployment.GEMINI_FLASH_1_5_V1
                | ChatCompletionDeployment.GEMINI_FLASH_1_5_V2
            ):
                return await Gemini_1_5_Prompt.parse(
                    self.file_storage, tools, messages
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
            tools = prompt.tools.to_gemini_tools()
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

    @staticmethod
    async def process_chunks(
        consumer: Consumer,
        tools: ToolsConfig,
        generator: Callable[[], AsyncIterator[GenerationResponse]],
    ) -> AsyncIterator[str]:

        async for chunk in generator():
            if log.isEnabledFor(DEBUG):
                chunk_str = json_dumps(chunk, excluded_keys=["safety_ratings"])
                log.debug(f"response chunk: {chunk_str}")

            if chunk.candidates:
                candidate = chunk.candidates[0]

                content = candidate.text
                await consumer.append_content(content)
                yield content

                await create_function_calls(candidate, consumer, tools)
                await create_attachments_from_citations(candidate, consumer)
                await set_finish_reason(candidate, consumer)

            if chunk.usage_metadata:
                await set_usage(chunk.usage_metadata, consumer)

            if chunk.prompt_feedback:
                await consumer.set_finish_reason(FinishReason.CONTENT_FILTER)

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
    ) -> Tuple[DiscardedMessages, GeminiPrompt]:
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


async def set_finish_reason(candidate: Candidate, consumer: Consumer) -> None:
    openai_reason = to_openai_finish_reason(
        finish_reason=candidate.finish_reason,
        retriable=consumer.is_empty(),
    )

    if openai_reason is not None:
        await consumer.set_finish_reason(openai_reason)


async def create_attachments_from_citations(
    candidate: Candidate, consumer: Consumer
) -> None:
    citation_metadata = candidate.citation_metadata

    if (
        citation_metadata is None
        or citation_metadata.citations is None
        or not len(citation_metadata.citations)
    ):
        return None

    for citation in citation_metadata.citations:
        if citation.uri:
            await consumer.add_attachment(
                Attachment(url=citation.uri, title=citation.title)
            )


async def set_usage(
    usage: GenerateContentResponse.UsageMetadata, consumer: Consumer
) -> None:
    log.debug(f"usage: {json_dumps(usage)}")
    await consumer.set_usage(
        TokenUsage(
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
        )
    )


async def create_function_calls(
    candidate: Candidate, consumer: Consumer, tools: ToolsConfig
) -> None:
    for call in candidate.function_calls:
        arguments = json.dumps(recurse_proto_marshal_to_dict(call.args))

        if tools.is_tool:
            id = tools.create_fresh_tool_call_id(call.name)
            log.debug(f"tool call: id={id}, {json_dumps(call)}")
            await consumer.create_tool_call(
                id=id,
                name=call.name,
                arguments=arguments,
            )
        else:
            log.debug(f"function call: {json_dumps(call)}")
            await consumer.create_function_call(
                name=call.name,
                arguments=arguments,
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

        # The following finish reasons could be usually fixed with a retry
        case GenFinishReason.OTHER:
            raise FinishReasonOtherError(
                msg="The model terminated generation unexpectedly",
                retriable=retriable,
            )
        case GenFinishReason.MALFORMED_FUNCTION_CALL:
            raise FinishReasonOtherError(
                msg="The function call generated by the model is invalid",
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
