from logging import DEBUG
from typing import List, Type, assert_never

from aidial_sdk.chat_completion import Message
from aidial_sdk.exceptions import InternalServerError
from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from anthropic._resource import AsyncAPIResource
from anthropic.lib.streaming import BetaInputJsonEvent as InputJsonEvent
from anthropic.lib.streaming import BetaTextEvent as TextEvent
from anthropic.lib.streaming import (
    ParsedBetaContentBlockStopEvent as ParsedContentBlockStopEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaCitationEvent as CitationEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaSignatureEvent as SignatureEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaThinkingEvent as ThinkingEvent,
)
from anthropic.lib.streaming._beta_types import (
    ParsedBetaMessageStopEvent as ParsedMessageStopEvent,
)
from anthropic.resources.beta import AsyncMessages as FirstPartyAsyncMessagesAPI
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import (
    BetaBashCodeExecutionToolResultBlock as BashCodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaCodeExecutionToolResultBlock as CodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaContainerUploadBlock as ContainerUploadBlock,
)
from anthropic.types.beta import BetaMCPToolResultBlock as MCPToolResultBlock
from anthropic.types.beta import BetaMCPToolUseBlock as MCPToolUseBlock
from anthropic.types.beta import (
    BetaRawContentBlockDeltaEvent as ContentBlockDeltaEvent,
)
from anthropic.types.beta import (
    BetaRawContentBlockStartEvent as ContentBlockStartEvent,
)
from anthropic.types.beta import BetaRawMessageDeltaEvent as MessageDeltaEvent
from anthropic.types.beta import BetaRawMessageStartEvent as MessageStartEvent
from anthropic.types.beta import (
    BetaRedactedThinkingBlock as RedactedThinkingBlock,
)
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import (
    BetaTextEditorCodeExecutionToolResultBlock as TextEditorCodeExecutionToolResultBlock,
)
from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock
from anthropic.types.beta import (
    BetaToolSearchToolResultBlock as ToolSearchToolResultBlock,
)
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock
from anthropic.types.beta import (
    BetaWebFetchToolResultBlock as WebFetchToolResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaTextBlock as ParsedTextBlock,
)
from pydantic import Field
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.claude.finish_reason import (
    to_dial_finish_reason,
)
from aidial_adapter_vertexai.chat.claude.output import (
    create_citations,
    process_tools_block,
)
from aidial_adapter_vertexai.chat.claude.params import (
    create_chat_params,
    none_to_omit,
)
from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.chat.claude.prompt.claude_3 import (
    parse_claude_3_prompt,
)
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
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.pydantic import ExtraForbidModel


# Beta AsyncMessages doesn't provide stream and count_tokens,
# so we enabled it via the adapter.
class _AsyncMessagesAdapter(AsyncAPIResource):
    create = FirstPartyAsyncMessagesAPI.create
    stream = FirstPartyAsyncMessagesAPI.stream
    count_tokens = FirstPartyAsyncMessagesAPI.count_tokens

    def __init__(self, resource: AsyncAPIResource):
        super().__init__(resource._client)


class ClaudeConfiguration(ExtraForbidModel):
    enable_citations: bool = False
    betas: List[AnthropicBetaParam] | None = Field(
        default=None,
        description="List of beta features to enable. Make sure to check if the given feature is supported by the Claude deployment you are using.",
    )


class ClaudeChatCompletionAdapter(ChatCompletionAdapter[ClaudePrompt]):
    deployment: AdapterDeployment[ClaudeDeployment]
    client: AsyncAnthropicVertex | AsyncAnthropic

    def __init__(
        self,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[ClaudeDeployment],
        client: AsyncAnthropicVertex | AsyncAnthropic,
    ):
        self.file_storage = file_storage
        self.deployment = deployment
        self.client = client

    @property
    def model_id(self) -> str:
        return self.deployment.upstream_deployment_id

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[ClaudeDeployment],
        config: UpstreamConfig,
    ) -> "ClaudeChatCompletionAdapter":
        client = await config.get_anthropic_client()
        return cls(file_storage, deployment, client)

    @override
    async def configuration(self) -> Type[ClaudeConfiguration]:
        return ClaudeConfiguration

    @override
    async def parse_prompt(
        self,
        params: ModelParameters,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: List[Message],
    ) -> ClaudePrompt | UserError:
        configuration = params.parse_configuration(await self.configuration())
        enable_citations = configuration.enable_citations

        static_tools.not_supported()
        match self.deployment.reference_deployment_id:
            case (
                ChatCompletionDeployment.CLAUDE_3_5_SONNET_V2
                | ChatCompletionDeployment.CLAUDE_3_OPUS
                | ChatCompletionDeployment.CLAUDE_3_5_SONNET
                | ChatCompletionDeployment.CLAUDE_3_HAIKU
                | ChatCompletionDeployment.CLAUDE_3_7_SONNET
                | ChatCompletionDeployment.CLAUDE_4_OPUS
                | ChatCompletionDeployment.CLAUDE_4_SONNET
                | ChatCompletionDeployment.CLAUDE_4_1_OPUS
                | ChatCompletionDeployment.CLAUDE_4_5_HAIKU
                | ChatCompletionDeployment.CLAUDE_4_5_SONNET
            ):
                return await parse_claude_3_prompt(
                    self.file_storage,
                    tools,
                    messages,
                    supports_vision=True,
                    enable_citations=enable_citations,
                )
            case ChatCompletionDeployment.CLAUDE_3_5_HAIKU:
                return await parse_claude_3_prompt(
                    self.file_storage,
                    tools,
                    messages,
                    supports_vision=False,
                    enable_citations=enable_citations,
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
        configuration = params.parse_configuration(await self.configuration())
        tools_mode = prompt.tools.tools_mode
        claude_params = create_chat_params(params, prompt, configuration.betas)

        async with _AsyncMessagesAdapter(self.client.beta.messages).stream(
            messages=prompt.claude_messages,
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
                    case ParsedContentBlockStopEvent(
                        content_block=content_block
                    ):
                        match content_block:
                            case ToolUseBlock():
                                await process_tools_block(
                                    consumer, content_block, tools_mode
                                )
                            case TextBlock(citations=citations):
                                # The text content is already handled in TextEvent handler.
                                for citation in citations or []:
                                    await create_citations(
                                        consumer, prompt, citation
                                    )
                            case ThinkingBlock() | RedactedThinkingBlock():
                                # thinking isn't yet supported
                                pass
                            case (
                                ServerToolUseBlock()
                                | WebSearchToolResultBlock()
                                | CodeExecutionToolResultBlock()
                                | MCPToolUseBlock()
                                | MCPToolResultBlock()
                                | ContainerUploadBlock()
                                | ParsedTextBlock()
                                | BashCodeExecutionToolResultBlock()
                                | TextEditorCodeExecutionToolResultBlock()
                            ):
                                log.error(
                                    f"Content block of type {content_block.type} isn't supported"
                                )
                            case _:
                                assert_never(content_block)
                    case ParsedMessageStopEvent(message=message):
                        stop_reason = message.stop_reason
                    case (
                        InputJsonEvent()
                        | ContentBlockStartEvent()
                        | ContentBlockDeltaEvent()
                    ):
                        pass
                    case ThinkingEvent() | SignatureEvent() | CitationEvent():
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
        self, params: ModelParameters, consumer: Consumer, prompt: ClaudePrompt
    ):
        configuration = params.parse_configuration(await self.configuration())
        tools_mode = prompt.tools.tools_mode
        claude_params = create_chat_params(params, prompt, configuration.betas)

        message = await self.client.beta.messages.create(
            messages=prompt.claude_messages,
            model=self.model_id,
            **claude_params,
            stream=False,
        )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(message)}")

        for content in message.content:
            match content:
                case TextBlock(text=text, citations=citations):
                    await consumer.append_content(text)
                    for citation in citations or []:
                        await create_citations(consumer, prompt, citation)
                case ToolUseBlock():
                    await process_tools_block(consumer, content, tools_mode)
                case ThinkingBlock() | RedactedThinkingBlock():
                    # thinking isn't yet supported
                    pass
                case (
                    ServerToolUseBlock()
                    | WebSearchToolResultBlock()
                    | CodeExecutionToolResultBlock()
                    | MCPToolUseBlock()
                    | MCPToolResultBlock()
                    | ContainerUploadBlock()
                    | BashCodeExecutionToolResultBlock()
                    | TextEditorCodeExecutionToolResultBlock()
                    | WebFetchToolResultBlock()
                    | ToolSearchToolResultBlock()
                ):
                    log.error(
                        f"Content block of type {content.type} isn't supported"
                    )
                case _:
                    assert_never(content)

        if not message.content:
            # Appending at least some content, otherwise it's not possible to report usage
            await consumer.append_content("")

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
        prompt = await prompt.truncate(
            tokenize=self.count_prompt_tokens, user_limit=max_prompt_tokens
        )

        return TruncatedPrompt(
            prompt=prompt, discarded_messages=prompt.removed_indices
        )

    @override
    async def count_prompt_tokens(self, prompt: ClaudePrompt) -> int:
        return (
            await _AsyncMessagesAdapter(self.client.beta.messages).count_tokens(
                model=self.model_id,
                messages=prompt.claude_messages,
                system=none_to_omit(prompt.system),
                tools=none_to_omit(prompt.tools.to_claude_tools()),
                tool_choice=none_to_omit(prompt.tools.to_claude_tool_choice()),
            )
        ).input_tokens

    @override
    async def count_completion_tokens(self, string: str) -> int:
        raise InternalServerError("Tokenization of strings is not supported")
