import json

from aidial_sdk.chat_completion import Attachment
from google.cloud.aiplatform_v1beta1.types.prediction_service import (
    GenerateContentResponse,
)
from vertexai.preview.generative_models import Candidate

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.gemini.grounding import (
    google_search_grounding_tokens,
)
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.deployments import GeminiDeployment
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.json import json_dumps
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log
from aidial_adapter_vertexai.utils.protobuf import recurse_proto_marshal_to_dict


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
    usage: GenerateContentResponse.UsageMetadata,
    consumer: Consumer,
    deployment: GeminiDeployment,
    is_grounding_added: bool = False,
) -> None:
    log.debug(f"usage: {json_dumps(usage)}")
    completion_tokens = usage.candidates_token_count or 0
    if is_grounding_added:
        completion_tokens += google_search_grounding_tokens(deployment)
    await consumer.set_usage(
        TokenUsage(
            prompt_tokens=usage.prompt_token_count or 0,
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
