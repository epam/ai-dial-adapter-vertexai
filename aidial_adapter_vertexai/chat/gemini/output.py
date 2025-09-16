import json
from logging import DEBUG
from urllib.parse import urlparse

from aidial_sdk.chat_completion import Attachment
from google.cloud.aiplatform_v1beta1.types.prediction_service import (
    GenerateContentResponse,
)
from google.genai.types import Candidate as GenAICandidate
from google.genai.types import (
    GenerateContentResponseUsageMetadata as GenAIUsageMetadata,
)
from google.genai.types import Part as GenAIPart
from vertexai.preview.generative_models import Candidate

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.gemini.grounding import (
    google_search_grounding_tokens,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import (
    GeminiDeployment,
    GeminiLegacyDeployment,
)
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.protobuf import recurse_proto_marshal_to_dict


def _is_absolute_url(uri: str) -> bool:
    parsed = urlparse(uri)
    return bool(parsed.scheme and parsed.netloc)


async def create_citations(
    candidate: Candidate | GenAICandidate, consumer: Consumer
) -> None:
    if (citation_metadata := candidate.citation_metadata) is None:
        return None

    for citation in citation_metadata.citations or []:
        if uri := citation.uri:
            if _is_absolute_url(uri):
                await consumer.add_attachment(
                    Attachment(
                        title=citation.title,
                        type="text/markdown",
                        url=uri,
                        reference_url=uri,
                        reference_type="text/markdown",
                    )
                )
            else:
                log.error(f"Gemini provided invalid citation URI: {uri!r}.")


async def set_usage(
    usage: GenerateContentResponse.UsageMetadata | GenAIUsageMetadata,
    consumer: Consumer,
    deployment: GeminiLegacyDeployment | GeminiDeployment,
    is_grounding_added: bool = False,
) -> None:
    if log.isEnabledFor(DEBUG):
        log.debug(f"usage: {json_dumps(usage)}")

    completion_tokens = usage.candidates_token_count or 0
    completion_tokens += usage.thoughts_token_count or 0

    if is_grounding_added:
        completion_tokens += google_search_grounding_tokens(deployment)

    await consumer.set_usage(
        TokenUsage(
            prompt_tokens=usage.prompt_token_count or 0,
            prompt_cached_tokens=usage.cached_content_token_count,
            completion_tokens=completion_tokens,
        )
    )


async def create_function_calls(
    candidate: Candidate, consumer: Consumer, tools: ToolsConfig
) -> None:
    for call in candidate.function_calls:
        arguments = json.dumps(recurse_proto_marshal_to_dict(call.args))

        if tools.is_tool:
            id = tools.create_fresh_tool_call_id(call.name)
            if log.isEnabledFor(DEBUG):
                log.debug(f"tool call: id={id}, {json_dumps(call)}")
            await consumer.create_tool_call(
                id=id,
                name=call.name,
                arguments=arguments,
            )
        else:
            if log.isEnabledFor(DEBUG):
                log.debug(f"function call: {json_dumps(call)}")
            await consumer.create_function_call(
                name=call.name,
                arguments=arguments,
            )


async def create_function_calls_from_genai(
    part: GenAIPart, consumer: Consumer, tools: ToolsConfig
) -> None:
    if not (function_call := part.function_call):
        return
    if not function_call.name:
        return

    function_args = (
        json.dumps(function_call.args) if function_call.args else None
    )
    if tools.is_tool:
        await consumer.create_tool_call(
            id=tools.create_fresh_tool_call_id(function_call.name),
            name=function_call.name,
            arguments=function_args,
        )
    else:
        if consumer.has_function_call:
            log.warning(
                "The model generated more than one tool call. "
                "Only the first one will be taken in to account."
            )
        else:
            await consumer.create_function_call(
                name=function_call.name,
                arguments=function_args,
            )
