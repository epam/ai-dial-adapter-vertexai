import json
from collections.abc import Sequence
from dataclasses import dataclass
from logging import DEBUG
from typing import assert_never

from aidial_sdk.chat_completion import FinishReason, Message
from mistralai.client import models as models
from mistralai.client.types import basemodel as models_base
from mistralai.gcp.client import models as gcp_models
from mistralai.gcp.client.types import basemodel as gcp_models_base
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.mistral.prompt import (
    MistralPrompt,
    MistralPromptParser,
)
from aidial_adapter_vertexai.chat.mistral.state import ToolCallState
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import MistralDeployment
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.storage import FileStorage
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.upstream_config import (
    MistralClient,
    UpstreamConfig,
)
from aidial_adapter_vertexai.utils.adapter_deployments import AdapterDeployment
from aidial_adapter_vertexai.utils.json import json_dumps_short
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.timer import Timer

AssistantContent = (
    str
    | list[models.ContentChunk]
    | list[gcp_models.ContentChunk]
    | models_base.Unset
    | gcp_models_base.Unset
    | None
)


@dataclass
class MistralChatCompletionAdapter(ChatCompletionAdapter[MistralPrompt]):
    file_storage: FileStorage | None
    deployment: AdapterDeployment[MistralDeployment]
    client: MistralClient

    @classmethod
    async def create(
        cls,
        file_storage: FileStorage | None,
        deployment: AdapterDeployment[MistralDeployment],
        config: UpstreamConfig,
    ) -> "MistralChatCompletionAdapter":
        client = await config.get_mistral_client()
        return cls(
            file_storage=file_storage, deployment=deployment, client=client
        )

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
    ) -> MistralPrompt:
        static_tools.not_supported()
        return await MistralPromptParser.parse(
            params, tools, self.file_storage, messages
        )

    @override
    async def chat(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: MistralPrompt,
    ) -> None:
        if log.isEnabledFor(DEBUG):
            request_str = json_dumps_short(
                {"parameters": params, "prompt": prompt}, exclude_none=True
            )
            log.debug(f"predict request: {request_str}")

        with Timer("predict timing: {time}", log.debug):
            if params.stream:
                usage = await self._chat_stream(params, consumer, prompt)
            else:
                usage = await self._chat_non_stream(params, consumer, prompt)

        if usage is not None:
            await consumer.set_usage(_to_token_usage(usage))

    async def _chat_non_stream(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: MistralPrompt,
    ) -> models.UsageInfo | gcp_models.UsageInfo | None:
        response = await self.client.chat.complete_async(
            model=self.model_id,
            messages=prompt.messages_unwrap,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop,
            random_seed=params.seed,
            response_format=prompt.response_format_unwrap,
            tools=prompt.tools_unwrap,
            tool_choice=prompt.tool_choice_unwrap,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
        )
        if log.isEnabledFor(DEBUG):
            log.debug(f"predict response: {json_dumps_short(response)}")

        usage = response.usage

        if not response.choices:
            return usage

        choice = response.choices[0]
        if choice.message is None:
            return usage

        content = _stringify_content(choice.message.content)
        if content:
            await consumer.append_content(content)

        allow_tool_calls = prompt.tools is not None
        if tool_calls := choice.message.tool_calls:
            await consume_tool_calls(
                tool_calls,
                consumer,
                use_tool_api=prompt.use_tool_api,
                allow_tool_calls=allow_tool_calls,
            )

        if (finish_reason := choice.finish_reason) is not None:
            await consumer.set_finish_reason(to_finish_reason(finish_reason))

        return usage

    async def _chat_stream(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: MistralPrompt,
    ) -> models.UsageInfo | gcp_models.UsageInfo | None:
        usage: models.UsageInfo | gcp_models.UsageInfo | None = None
        tool_calls_state: dict[int, ToolCallState] = {}
        tool_calls_emitted = False
        allow_tool_calls = prompt.tools is not None

        stream = await self.client.chat.stream_async(
            model=self.model_id,
            messages=prompt.messages_unwrap,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop,
            random_seed=params.seed,
            response_format=prompt.response_format_unwrap,
            tools=prompt.tools_unwrap,
            tool_choice=prompt.tool_choice_unwrap,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
        )
        async with stream as events:
            async for event in events:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"stream event: {json_dumps_short(event)}")

                if event.data.usage is not None:
                    usage = event.data.usage

                finish_reason = await consume_stream_chunk(
                    event.data,
                    consumer,
                    tool_calls_state,
                    allow_tool_calls=allow_tool_calls,
                )
                if (
                    allow_tool_calls
                    and not tool_calls_emitted
                    and finish_reason == FinishReason.TOOL_CALLS
                ):
                    streamed_tool_calls = [
                        state.to_tool_call()
                        for state in tool_calls_state.values()
                    ]
                    await consume_tool_calls(
                        streamed_tool_calls,
                        consumer,
                        use_tool_api=prompt.use_tool_api,
                        allow_tool_calls=True,
                    )
                    tool_calls_emitted = True

        return usage


async def consume_stream_chunk(
    chunk: models.CompletionChunk | gcp_models.CompletionChunk,
    consumer: Consumer,
    tool_calls_state: dict[int, ToolCallState],
    *,
    allow_tool_calls: bool,
) -> FinishReason | None:
    if not chunk.choices:
        return None

    choice = chunk.choices[0]
    raw_content = choice.delta.content if choice.delta else None
    match raw_content:
        case None | models_base.Unset() | gcp_models_base.Unset():
            content = ""
        case str():
            content = raw_content
        case list():
            chunks: list[str] = []
            for item in raw_content:
                match item:
                    case models.TextChunk(text=text):
                        chunks.append(text)
                    case _:
                        log.warning(
                            f"Ignoring content chunk of type {type(item).__name__}; expected TextChunk."
                        )
            content = "".join(chunks)
        case _:
            content = str(raw_content)
    if content:
        await consumer.append_content(content)

    if choice.delta and choice.delta.tool_calls:
        append_tool_calls_state(tool_calls_state, choice.delta.tool_calls)

    if choice.finish_reason is not None:
        finish_reason = to_finish_reason(str(choice.finish_reason))
        if not allow_tool_calls and finish_reason == FinishReason.TOOL_CALLS:
            finish_reason = FinishReason.STOP
        await consumer.set_finish_reason(finish_reason)
        return finish_reason
    return None


def to_finish_reason(reason: str) -> FinishReason:
    if reason in ("length", "model_length"):
        return FinishReason.LENGTH
    if reason == "tool_calls":
        return FinishReason.TOOL_CALLS
    if reason == "error":
        return FinishReason.CONTENT_FILTER
    return FinishReason.STOP


def _to_token_usage(
    usage: models.UsageInfo | gcp_models.UsageInfo,
) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )


def _stringify_content(content: AssistantContent) -> str:
    match content:
        case str():
            return content
        case None | models_base.Unset() | gcp_models_base.Unset():
            return ""
        case list():
            chunks: list[str] = []
            for item in content:
                match item:
                    case models.TextChunk(text=text):
                        chunks.append(text)
                    case _:
                        log.warning(
                            f"Ignoring content chunk of type {type(item).__name__}; expected TextChunk."
                        )
            return "".join(chunks)
        case _:
            assert_never(content)


async def consume_tool_calls(
    tool_calls: list[models.ToolCall] | list[gcp_models.ToolCall],
    consumer: Consumer,
    *,
    use_tool_api: bool,
    allow_tool_calls: bool,
) -> None:
    if not allow_tool_calls:
        log.warning("Ignoring tool calls from model when tools are undeclared")
        return

    for call in tool_calls:
        name = call.function.name
        if not name:
            continue

        arguments = _stringify_arguments(call.function.arguments)
        if use_tool_api:
            await consumer.create_tool_call(call.id or "", name, arguments)
            continue

        if consumer.has_function_call:
            log.warning(
                "The model generated more than one function call. "
                "Only the first one will be taken into account."
            )
            continue

        await consumer.create_function_call(name, arguments)


def append_tool_calls_state(
    state: dict[int, ToolCallState],
    tool_calls: Sequence[models.ToolCall | gcp_models.ToolCall],
) -> None:
    for call in tool_calls:
        idx = call.index or 0
        current = state.get(idx)
        if current is None:
            current = state[idx] = ToolCallState(index=idx)

        if call.id and call.id != "null":
            current.id = call.id

        if call.function.name:
            current.name = call.function.name

        current.arguments += _stringify_arguments(call.function.arguments)


def _stringify_arguments(
    arguments: models.Arguments | gcp_models.Arguments,
) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"))
