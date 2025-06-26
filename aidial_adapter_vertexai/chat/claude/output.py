import json
from typing import assert_never

from anthropic.types.beta import (
    BetaCitationCharLocation as CitationCharLocation,
)
from anthropic.types.beta import (
    BetaCitationContentBlockLocation as CitationContentBlockLocation,
)
from anthropic.types.beta import (
    BetaCitationPageLocation as CitationPageLocation,
)
from anthropic.types.beta import (
    BetaCitationsWebSearchResultLocation as CitationsWebSearchResultLocation,
)
from anthropic.types.beta import BetaTextCitation as TextCitation
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock

from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudePrompt
from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.chat.tools import ToolsMode
from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log


async def process_tools_block(
    consumer: Consumer, block: ToolUseBlock, tools_mode: ToolsMode | None
):
    match tools_mode:
        case ToolsMode.TOOLS:
            await consumer.create_tool_call(
                id=block.id,
                name=block.name,
                arguments=json.dumps(block.input),
            )
        case ToolsMode.FUNCTIONS:
            if consumer.has_function_call:
                log.warning(
                    "The model generated more than one tool call. "
                    "Only the first one will be taken in to account."
                )
            else:
                await consumer.create_function_call(
                    name=block.name,
                    arguments=json.dumps(block.input),
                )
        case None:
            raise ValidationError(
                "A model has called a tool, but no tools were given to the model in the first place."
            )
        case _:
            assert_never(tools_mode)


async def _add_document_citation(
    consumer: Consumer, prompt: ClaudePrompt, document_index: int
):
    resource = prompt.get_dial_resource(document_index)
    attachment = None if resource is None else resource.to_attachment()

    # NOTE: multiple citations to the same document are merged into one citation
    # until we find a better API to handle citations embedded in text.
    display_index = await consumer.add_citation_attachment(
        document_id=document_index, document=attachment
    )

    # NOTE: avoid adding citation URLs into the generated content,
    # since such references aren't easily portable (e.g. when a conversion is duplicated).
    await consumer.append_content(f"[{display_index}]")


async def create_citations(
    consumer: Consumer, prompt: ClaudePrompt, citation: TextCitation
):
    match citation:
        case CitationCharLocation(document_index=document_index):
            await _add_document_citation(consumer, prompt, document_index)

        case CitationPageLocation(document_index=document_index):
            await _add_document_citation(consumer, prompt, document_index)

        # custom document aren't supported yet
        case CitationContentBlockLocation():
            pass
        # web search isn't supported yet
        case CitationsWebSearchResultLocation():
            pass
        case _:
            assert_never(citation)
