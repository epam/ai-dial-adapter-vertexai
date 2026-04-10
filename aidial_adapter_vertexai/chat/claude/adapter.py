from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

from aidial_adapter_anthropic.adapter import (
    ChatCompletionAdapter as AnthropicClaudeAdapter,
)
from aidial_adapter_anthropic.adapter.claude import (
    create_adapter as anthropic_create_adapter,
)
from aidial_adapter_anthropic.dial.consumer import Consumer as AnthropicConsumer
from aidial_adapter_anthropic.dial.consumer import ToolUseMessage
from aidial_adapter_anthropic.dial.request import (
    ModelParameters as AnthropicModelParameters,
)
from aidial_adapter_anthropic.dial.storage import (
    FileStorage as AnthropicFileStorage,
)
from aidial_adapter_anthropic.dial.token_usage import (
    TokenUsage as AnthropicTokenUsage,
)
from aidial_adapter_anthropic.dial.tools import (
    ToolsConfig as AnthropicToolsConfig,
)
from aidial_adapter_anthropic.dial.tools import ToolsMode as AnthropicToolsMode
from aidial_sdk.chat_completion import (
    Attachment,
    FinishReason,
    FunctionCall,
    Message,
    ToolCall,
)
from pydantic import BaseModel

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig, ToolsMode
from aidial_adapter_vertexai.chat.truncate_prompt import TruncatedPrompt
from aidial_adapter_vertexai.deployments import ClaudeDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.upstream_config import UpstreamConfig
from aidial_adapter_vertexai.utils.adapter_deployment import AdapterDeployment
from aidial_adapter_vertexai.utils.env import get_env_int
from aidial_adapter_vertexai.utils.list import omit_by_indices

_CLAUDE_DEFAULT_MAX_TOKENS = get_env_int("CLAUDE_DEFAULT_MAX_TOKENS", 1536)


def _to_tools_mode(c: ToolsMode) -> AnthropicToolsMode:
    return (
        AnthropicToolsMode.FUNCTIONS
        if c == ToolsMode.FUNCTIONS
        else AnthropicToolsMode.TOOLS
    )


def _to_tool_config(c: ToolsConfig) -> AnthropicToolsConfig:
    return AnthropicToolsConfig(
        tools=c.tools,
        tools_mode=_to_tools_mode(c.tools_mode),
        tool_choice=c.tool_choice,
        tool_ids=c.tool_ids,
    )


def _to_model_params(
    params: ModelParameters, c: ToolsConfig
) -> AnthropicModelParameters:
    return AnthropicModelParameters(
        temperature=params.temperature,
        top_p=params.top_p,
        n=params.n or 1,
        stop=params.stop or [],
        seed=params.seed,
        max_tokens=params.max_tokens,
        max_prompt_tokens=params.max_prompt_tokens,
        stream=params.stream,
        tool_config=_to_tool_config(c),
        configuration=params.configuration,
    )


@dataclass
class _ConsumerAdapter(AnthropicConsumer):
    consumer: Consumer
    tool_calls: list[ToolUseMessage] = field(default_factory=list)

    def fork(self):
        return _ConsumerAdapter(self.consumer.fork())

    @property
    def choice(self):
        return self.consumer.choice

    async def close_content(self, finish_reason: FinishReason | None = None):
        if finish_reason is not None:
            await self.consumer.set_finish_reason(finish_reason)

    async def append_content(self, content: str):
        await self.consumer.append_content(content)

    async def add_attachment(self, attachment: Attachment):
        await self.consumer.add_attachment(attachment)

    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int:
        return await self.consumer.add_citation_attachment(
            document_id, document
        )

    async def add_usage(self, usage: AnthropicTokenUsage):
        await self.consumer.set_usage(
            TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                prompt_cached_tokens=usage.cache_read_input_tokens,
            )
        )

    async def set_discarded_messages(
        self, discarded_messages: list[int] | None
    ):
        # Discarded message are tracked on the top-level
        pass

    async def get_discarded_messages(self) -> list[int] | None:
        # Discarded message are tracked on the top-level
        pass

    async def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        tool_call = ToolUseMessage(
            call=self.choice.create_function_tool_call(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            ),
            snapshot=call.function.arguments,
        )
        self.tool_calls.append(tool_call)
        return tool_call

    async def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        tool_call = ToolUseMessage(
            call=self.choice.create_function_call(
                name=call.name,
                arguments=call.arguments,
            ),
            snapshot=call.arguments,
        )
        self.tool_calls.append(tool_call)
        return tool_call

    @property
    def has_function_call(self) -> bool:
        return self.consumer.has_function_call

    async def __aenter__(self) -> _ConsumerAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        for tool_call in self.tool_calls:
            tool_call.close()

        return self.consumer.__exit__(exc_type, exc, traceback)


@dataclass
class ClaudePrompt:
    params: AnthropicModelParameters
    messages: list[Message]


@dataclass
class ClaudeChatCompletionAdapter(ChatCompletionAdapter[ClaudePrompt]):
    claude_adapter: AnthropicClaudeAdapter

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[ClaudeDeployment],
        config: UpstreamConfig,
    ) -> ClaudeChatCompletionAdapter:
        if file_storage is not None:
            storage = AnthropicFileStorage(
                dial_url=file_storage.dial_url,
                api_key=file_storage.api_key,
            )
        else:
            storage = None

        client = await config.get_anthropic_client()
        claude_adapter = await anthropic_create_adapter(
            deployment=deployment.upstream_deployment_id,
            storage=storage,
            client=client,
            default_max_tokens=_CLAUDE_DEFAULT_MAX_TOKENS,
            supports_documents=True,
            supports_thinking=True,
        )
        return cls(claude_adapter=claude_adapter)

    async def parse_prompt(
        self,
        params: ModelParameters,
        tools: ToolsConfig,
        static_tools: StaticToolsConfig,
        messages: list[Message],
    ) -> ClaudePrompt | UserError:
        static_tools.not_supported()
        model_params = _to_model_params(params, tools)
        return ClaudePrompt(model_params, messages)

    async def chat(
        self, params: ModelParameters, consumer: Consumer, prompt: ClaudePrompt
    ) -> None:
        await self.claude_adapter.chat(
            _ConsumerAdapter(consumer), prompt.params, prompt.messages
        )

    async def configuration(self) -> type[BaseModel] | None:
        return await self.claude_adapter.configuration()

    async def truncate_prompt(
        self, prompt: ClaudePrompt, max_prompt_tokens: int
    ) -> TruncatedPrompt[ClaudePrompt]:
        discarded_indices = (
            await self.claude_adapter.compute_discarded_messages(
                prompt.params, prompt.messages
            )
        ) or []
        prompt.messages = omit_by_indices(prompt.messages, discarded_indices)
        return TruncatedPrompt(
            prompt=prompt,
            discarded_messages=discarded_indices,
        )

    async def count_prompt_tokens(self, prompt: ClaudePrompt) -> int:
        return await self.claude_adapter.count_prompt_tokens(
            prompt.params, prompt.messages
        )

    async def count_completion_tokens(self, string: str) -> int:
        return await self.claude_adapter.count_completion_tokens(string)
