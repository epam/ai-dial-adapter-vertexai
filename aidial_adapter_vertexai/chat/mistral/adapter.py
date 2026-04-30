import json
from dataclasses import dataclass
from logging import DEBUG
from typing import Any, Literal, cast

from aidial_sdk.chat_completion import FinishReason, Message
from mistralai.gcp.client.models import (
    CompletionChunk,
    ToolCall,
    UsageInfo,
)
from typing_extensions import override

from aidial_adapter_vertexai.chat.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.mistral.prompt import (
    MistralPrompt,
    MistralPromptParser,
)
from aidial_adapter_vertexai.chat.mistral.state import _ToolCallState
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
                {
                    "model": self.model_id,
                    "stream": params.stream,
                    "messages": prompt.messages,
                    "has_tools": prompt.tools is not None,
                    "tool_choice": prompt.tool_choice,
                    "response_format": prompt.response_format,
                },
                exclude_none=True,
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
    ) -> UsageInfo | None:
        response = await self.client.chat.complete_async(
            model=self.model_id,
            messages=prompt.messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop,
            random_seed=params.seed,
            # The upstream client can be either `Mistral` or `MistralGCP`.
            # Their generated model classes are structurally compatible but
            # live in different modules, so pyright treats them as distinct.
            response_format=cast(Any, prompt.response_format),
            tools=cast(Any, prompt.tools),
            tool_choice=cast(Any, prompt.tool_choice),
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            prompt_mode=_to_prompt_mode(params),
        )
        if log.isEnabledFor(DEBUG):
            log.debug(f"predict response: {json_dumps_short(response)}")

        if not response.choices:
            return cast(UsageInfo | None, response.usage)

        choice = response.choices[0]
        if choice.message is None:
            return cast(UsageInfo | None, response.usage)

        content = _stringify_content(choice.message.content)
        if content:
            await consumer.append_content(content)

        allow_tool_calls = prompt.tools is not None
        await _consume_tool_calls(
            choice.message.tool_calls,
            consumer,
            use_tool_api=prompt.use_tool_api,
            allow_tool_calls=allow_tool_calls,
        )

        if choice.finish_reason is not None:
            finish_reason = _to_finish_reason(str(choice.finish_reason))
            if (
                not allow_tool_calls
                and finish_reason == FinishReason.TOOL_CALLS
            ):
                # When no tools are declared, ignore model-hallucinated tool calls
                # and preserve OpenAI-compatible graceful fallback behavior.
                finish_reason = FinishReason.STOP
            await consumer.set_finish_reason(finish_reason)

        return cast(UsageInfo | None, response.usage)

    async def _chat_stream(
        self,
        params: ModelParameters,
        consumer: Consumer,
        prompt: MistralPrompt,
    ) -> UsageInfo | None:
        usage = None
        tool_calls_state: dict[int, _ToolCallState] = {}
        tool_calls_emitted = False
        allow_tool_calls = prompt.tools is not None

        stream = await self.client.chat.stream_async(
            model=self.model_id,
            messages=prompt.messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop,
            random_seed=params.seed,
            # The upstream client can be either `Mistral` or `MistralGCP`.
            # Their generated model classes are structurally compatible but
            # live in different modules, so pyright treats them as distinct.
            response_format=cast(Any, prompt.response_format),
            tools=cast(Any, prompt.tools),
            tool_choice=cast(Any, prompt.tool_choice),
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            prompt_mode=_to_prompt_mode(params),
        )
        async with stream as events:
            async for event in events:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"stream event: {json_dumps_short(event)}")
                if event.data.usage is not None:
                    usage = event.data.usage

                finish_reason = await _consume_stream_chunk(
                    cast(CompletionChunk, event.data),
                    consumer,
                    tool_calls_state,
                    allow_tool_calls=allow_tool_calls,
                )
                if (
                    allow_tool_calls
                    and not tool_calls_emitted
                    and finish_reason == FinishReason.TOOL_CALLS
                ):
                    await _consume_tool_calls(
                        [
                            state.to_tool_call()
                            for state in tool_calls_state.values()
                        ],
                        consumer,
                        use_tool_api=prompt.use_tool_api,
                        allow_tool_calls=True,
                    )
                    tool_calls_emitted = True

        return cast(UsageInfo | None, usage)


async def _consume_stream_chunk(
    chunk: CompletionChunk,
    consumer: Consumer,
    tool_calls_state: dict[int, _ToolCallState],
    *,
    allow_tool_calls: bool,
) -> FinishReason | None:
    if not chunk.choices:
        return None

    choice = chunk.choices[0]
    content = _stringify_content(choice.delta.content if choice.delta else None)
    if content:
        await consumer.append_content(content)

    if choice.delta and choice.delta.tool_calls:
        _append_tool_calls_state(tool_calls_state, choice.delta.tool_calls)

    if choice.finish_reason is not None:
        finish_reason = _to_finish_reason(str(choice.finish_reason))
        if not allow_tool_calls and finish_reason == FinishReason.TOOL_CALLS:
            finish_reason = FinishReason.STOP
        await consumer.set_finish_reason(finish_reason)
        return finish_reason
    return None


def _to_finish_reason(reason: str) -> FinishReason:
    if reason in ("length", "model_length"):
        return FinishReason.LENGTH
    if reason == "tool_calls":
        return FinishReason.TOOL_CALLS
    if reason == "error":
        return FinishReason.CONTENT_FILTER
    return FinishReason.STOP


def _to_token_usage(usage: UsageInfo) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)
    return str(content)


def _to_prompt_mode(params: ModelParameters) -> Literal["reasoning"] | None:
    if params.reasoning_effort is None:
        return None
    if params.reasoning_effort.value == "none":
        return None
    return "reasoning"


async def _consume_tool_calls(
    tool_calls: Any,
    consumer: Consumer,
    *,
    use_tool_api: bool,
    allow_tool_calls: bool,
) -> None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return
    if not allow_tool_calls:
        log.warning("Ignoring tool calls from model when tools are undeclared")
        return

    for idx, call in enumerate(tool_calls):
        name = call.function.name
        if not name:
            continue

        arguments = _stringify_arguments(call.function.arguments)
        if use_tool_api:
            call_id = call.id if isinstance(call.id, str) else None
            # Some Mistral responses may omit an id or provide "null".
            # Keep upstream IDs when usable; otherwise generate deterministic
            # OpenAI-compatible fallback IDs.
            tool_call_id = (
                call_id
                if call_id not in (None, "", "null")
                else f"{name}_{idx + 1}"
            )
            await consumer.create_tool_call(tool_call_id, name, arguments)
            continue

        if consumer.has_function_call:
            log.warning(
                "The model generated more than one function call. "
                "Only the first one will be taken into account."
            )
            continue

        await consumer.create_function_call(name, arguments)


def _append_tool_calls_state(
    state: dict[int, _ToolCallState], tool_calls: list[ToolCall]
) -> None:
    for call in tool_calls:
        idx = call.index or 0
        current = state.get(idx)
        if current is None:
            current = state[idx] = _ToolCallState(index=idx)

        if call.id and call.id != "null":
            current.id = call.id

        if call.function.name:
            current.name = call.function.name

        current.arguments += _stringify_arguments(call.function.arguments)


def _stringify_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"))
