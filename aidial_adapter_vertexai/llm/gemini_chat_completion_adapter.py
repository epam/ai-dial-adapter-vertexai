from logging import DEBUG
from typing import AsyncIterator, Dict, List, Optional, Tuple

from aidial_sdk.chat_completion import Message
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
from typing_extensions import override
from vertexai.preview.generative_models import (
    ChatSession,
    GenerationConfig,
    GenerativeModel,
)

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.consumer import Consumer
from aidial_adapter_vertexai.llm.exceptions import UserError
from aidial_adapter_vertexai.llm.gemini_prompt import GeminiPrompt
from aidial_adapter_vertexai.llm.vertex_ai import (
    get_gemini_model,
    init_vertex_ai,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.storage import FileStorage
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer

HarmCategory = gapic_content_types.HarmCategory
HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold

BLOCK_NONE_SAFETY_SETTINGS: Dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=[params.stop]
        if isinstance(params.stop, str)
        else params.stop,
        top_p=params.top_p,
        # NOTE: param.n is currently emulated via multiple requests
        candidate_count=1,
    )


class GeminiChatCompletionAdapter(ChatCompletionAdapter[GeminiPrompt]):
    def __init__(
        self,
        file_storage: Optional[FileStorage],
        model: GenerativeModel,
        is_vision_model: bool,
    ):
        self.file_storage = file_storage
        self.model = model
        self.is_vision_model = is_vision_model

    @override
    async def parse_prompt(
        self, messages: List[Message]
    ) -> GeminiPrompt | UserError:
        if self.is_vision_model:
            return await GeminiPrompt.parse_vision(self.file_storage, messages)
        else:
            return GeminiPrompt.parse_non_vision(messages)

    @override
    async def truncate_prompt(
        self, prompt: GeminiPrompt, max_prompt_tokens: int
    ) -> Tuple[GeminiPrompt, int]:
        raise NotImplementedError(
            "Prompt truncation is not supported for Genimi model yet"
        )

    async def send_message_async(
        self, params: ModelParameters, prompt: GeminiPrompt
    ) -> AsyncIterator[str]:
        session = ChatSession(
            model=self.model, history=prompt.history, raise_on_blocked=False
        )
        parameters = create_generation_config(params)

        if params.stream:
            try:
                response = await session._send_message_streaming_async(
                    content=prompt.prompt,  # type: ignore
                    generation_config=parameters,
                    safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                    tools=None,
                )

                async for chunk in response:
                    yield chunk.text
            except Exception as e:
                log.debug(f"streaming request failed: {e}")
                log.exception("streaming request failed")
                raise
        else:
            response = await session._send_message_async(
                content=prompt.prompt,  # type: ignore
                generation_config=parameters,
                safety_settings=BLOCK_NONE_SAFETY_SETTINGS,
                tools=None,
            )

            yield response.text

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

            async for chunk in self.send_message_async(params, prompt):
                completion += chunk
                await consumer.append_content(chunk)

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
        is_vision_model: bool,
        project_id: str,
        location: str,
    ) -> "GeminiChatCompletionAdapter":
        await init_vertex_ai(project_id, location)
        model = await get_gemini_model(model_id)
        return cls(file_storage, model, is_vision_model)
