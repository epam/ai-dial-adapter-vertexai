from abc import abstractmethod
from typing import AsyncIterator, List, Tuple

from aidial_sdk.chat_completion import FinishReason, Message
from typing_extensions import override
from vertexai.preview.language_models import (
    ChatModel,
    CodeChatModel,
    CountTokensResponse,
)

from aidial_adapter_vertexai.chat.bison.prompt import BisonPrompt
from aidial_adapter_vertexai.chat.bison.truncate_prompt import (
    get_discarded_messages,
)
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer

BisonChatModel = ChatModel | CodeChatModel


class BisonChatCompletionAdapter(ChatCompletionAdapter[BisonPrompt]):
    def __init__(self, model: BisonChatModel):
        self.model = model

    @abstractmethod
    def send_message_async(
        self, params: ModelParameters, prompt: BisonPrompt
    ) -> AsyncIterator[str]:
        pass

    @override
    async def parse_prompt(self, messages: List[Message]) -> BisonPrompt:
        return BisonPrompt.parse(messages)

    @override
    async def truncate_prompt(
        self, prompt: BisonPrompt, max_prompt_tokens: int
    ) -> Tuple[BisonPrompt, List[int]]:
        return await get_discarded_messages(self, prompt, max_prompt_tokens)

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: BisonPrompt
    ) -> None:
        prompt_tokens = await self.count_prompt_tokens(prompt)

        with Timer("predict timing: {time}", log.debug):
            log.debug(
                "predict request: "
                f"parameters=({params}), "
                f"prompt=({prompt})"
            )

            completion = ""

            async for chunk in self.send_message_async(params, prompt):
                completion += chunk
                await consumer.append_content(chunk)

            log.debug(f"predict response: {completion!r}")

        completion_tokens = await self.count_completion_tokens(completion)

        # PaLM models do not return finish reason.
        # Use the heuristic to estimate it.
        if completion_tokens == params.max_tokens:
            await consumer.set_finish_reason(FinishReason.LENGTH)

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @override
    async def count_prompt_tokens(self, prompt: BisonPrompt) -> int:
        chat_session = self.model.start_chat(
            context=prompt.context, message_history=prompt.history
        )

        with Timer("count_tokens[prompt] timing: {time}", log.debug):
            resp = chat_session.count_tokens(message=prompt.user_prompt)
            log.debug(
                f"count_tokens[prompt] response: {_display_token_count(resp)}"
            )
            return resp.total_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        with Timer("count_tokens[completion] timing: {time}", log.debug):
            resp = self.model.start_chat().count_tokens(message=string)
            log.debug(
                f"count_tokens[completion] response: {_display_token_count(resp)}"
            )
            return resp.total_tokens


def _display_token_count(response: CountTokensResponse) -> str:
    return f"tokens: {response.total_tokens}, billable characters: {response.total_billable_characters}"
