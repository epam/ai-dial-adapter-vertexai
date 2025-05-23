import json
from typing import assert_never

from anthropic.types import (
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationsWebSearchResultLocation,
    TextCitation,
    ToolUseBlock,
)

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


async def create_attachments_from_citations(
    consumer: Consumer, citation: TextCitation
):
    match citation:
        case CitationCharLocation(document_index=document_index):
            document_url = "https://example.com"  # FIXME
            await consumer.append_content(
                f"[[{document_index}]({document_url})]"
            )
        case CitationPageLocation(
            document_index=document_index, start_page_number=start_page_number
        ):
            document_url = "https://example.com"  # FIXME
            await consumer.append_content(
                f"[[{document_index}]({document_url}#page={start_page_number})]"
            )
        # custom document aren't supported yet
        case CitationContentBlockLocation():
            pass
        # web search isn't supported yet
        case CitationsWebSearchResultLocation():
            pass
        case _:
            assert_never(citation)
