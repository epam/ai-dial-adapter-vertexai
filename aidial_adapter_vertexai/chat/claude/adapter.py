from logging import DEBUG
from typing import List, assert_never

from aidial_sdk.chat_completion import Message
from anthropic import AsyncAnthropicVertex, MessageStopEvent
from anthropic.lib.streaming import (
    ContentBlockStopEvent,
    InputJsonEvent,
    TextEvent,
)
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageDeltaEvent,
)
from anthropic.types import MessageParam as ClaudeMessage
from anthropic.types import MessageStartEvent, TextBlock, ToolUseBlock
from typing_extensions import override

from aidial_adapter_vertexai.app_config import ANTHROPIC_CLIENT
from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.claude.finish_reason import (
    to_dial_finish_reason,
)
from aidial_adapter_vertexai.chat.claude.params import (
    create_chat_params,
    none_to_not_given,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import (
    ClaudeConversation,
    ClaudePrompt,
)
from aidial_adapter_vertexai.chat.claude.prompt.claude_3 import Claude_3_Prompt
from aidial_adapter_vertexai.chat.claude.tools import process_tools_block
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    ClaudeDeployment,
)
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log


class ClaudeChatCompletionAdapter(ChatCompletionAdapter[ClaudePrompt]):
    deployment: ClaudeDeployment
    client: AsyncAnthropicVertex

    def __init__(
        self,
        file_storage: FileStorage | None,
        model_id: str,
        deployment: ClaudeDeployment,
    ):
        self.file_storage = file_storage
        self.model_id = model_id
        self.deployment = deployment
        self.client = ANTHROPIC_CLIENT

    @override
    async def parse_prompt(
        self,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> ClaudePrompt | UserError:

        static_tools.not_supported()
        match self.deployment:
            case (
                ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2
                | ChatCompletionDeployment.CLAUDE_3_5_HAIKU
                | ChatCompletionDeployment.CLAUDE_3_OPUS
                | ChatCompletionDeployment.CLAUDE_3_5_SONNET
                | ChatCompletionDeployment.CLAUDE_3_HAIKU
                | ChatCompletionDeployment.CLAUDE_3_SONNET
            ):
                return await Claude_3_Prompt.parse(
                    self.file_storage, tools, messages
                )
            case _:
                assert_never(self.deployment)

    @override
    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: ClaudePrompt
    ) -> None:

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {
                    "deployment": self.deployment,
                    "params": params,
                    "prompt": prompt,
                }
            )
            log.debug(f"request: {msg}")

        if params.stream:
            await self._invoke_streaming(params, consumer, prompt)
        else:
            await self._invoke_non_streaming(params, consumer, prompt)

    async def _invoke_streaming(
        self, params: ModelParameters, consumer: Consumer, prompt: ClaudePrompt
    ):
        tools_mode = prompt.tools.tools_mode
        claude_params = create_chat_params(params, prompt)

        async with self.client.messages.stream(
            messages=prompt.messages,
            model=self.model_id,
            **claude_params,
        ) as stream:
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = None
            async for event in stream:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"response event: {json_dumps_short(event)}")

                match event:
                    case MessageStartEvent(message=message):
                        prompt_tokens += message.usage.input_tokens
                    case TextEvent(text=text):
                        await consumer.append_content(text)
                    case MessageDeltaEvent(usage=usage):
                        completion_tokens += usage.output_tokens
                    case ContentBlockStopEvent(content_block=content_block):
                        match content_block:
                            case ToolUseBlock():
                                await process_tools_block(
                                    consumer, content_block, tools_mode
                                )
                            case TextBlock():
                                # Already handled in TextEvent
                                pass
                            case _:
                                assert_never(content_block)
                    case MessageStopEvent(message=message):
                        completion_tokens += message.usage.output_tokens
                        stop_reason = message.stop_reason
                    case (
                        InputJsonEvent()
                        | ContentBlockStartEvent()
                        | ContentBlockDeltaEvent()
                    ):
                        pass
                    case _:
                        assert_never(event)

            await consumer.set_finish_reason(
                to_dial_finish_reason(stop_reason, tools_mode)
            )

            await consumer.set_usage(
                TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

    async def _invoke_non_streaming(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: ClaudePrompt,
    ):
        tools_mode = prompt.tools.tools_mode
        claude_params = create_chat_params(params, prompt)

        message = await self.client.messages.create(
            messages=prompt.messages,
            model=self.model_id,
            **claude_params,
            stream=False,
        )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(message)}")

        for content in message.content:
            match content:
                case TextBlock(text=text):
                    await consumer.append_content(text)
                case ToolUseBlock():
                    await process_tools_block(consumer, content, tools_mode)
                case _:
                    assert_never(content)

        await consumer.set_finish_reason(
            to_dial_finish_reason(message.stop_reason, tools_mode)
        )

        await consumer.set_usage(
            TokenUsage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
            )
        )

    @override
    async def truncate_prompt(
        self, prompt: ClaudePrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[ClaudePrompt]:
        return await prompt.truncate(
            tokenizer=self.count_prompt_tokens, user_limit=max_prompt_tokens
        )

    @override
    async def count_prompt_tokens(self, prompt: ClaudePrompt) -> int:
        return (
            await self.client.messages.count_tokens(
                model=self.model_id,
                messages=prompt.messages,
                system=none_to_not_given(prompt.system),
                tools=none_to_not_given(prompt.tools.to_claude_tools()),
                tool_choice=none_to_not_given(
                    prompt.tools.to_claude_tool_config()
                ),
            )
        ).input_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        # FIXME: figure out if it's correct - add an integration test for it
        message: ClaudeMessage = {"role": "user", "content": string}
        return await self.count_prompt_tokens(
            ClaudePrompt(
                conversation=ClaudeConversation(system=None, messages=[message])
            )
        )

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        model_id: str,
        deployment: ClaudeDeployment,
    ) -> "ClaudeChatCompletionAdapter":
        return cls(file_storage, model_id, deployment)
