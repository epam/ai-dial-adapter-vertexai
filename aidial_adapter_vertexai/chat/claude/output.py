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
    consumer: Consumer,
    prompt: ClaudePrompt,
    document_index: int,
    extra_url: str = "",
):
    resource = prompt.get_dial_resource(document_index)
    if resource is None:
        return await consumer.append_content(f"[{document_index}]")

    attachment = resource.to_attachment()
    attachment.title = f"[{document_index}] {attachment.title or ''}".strip()

    if url := attachment.url:
        await consumer.append_content(f"[[{document_index}]({url}{extra_url})]")

    await consumer.add_attachment(attachment)


async def create_attachments_from_citations(
    consumer: Consumer, prompt: ClaudePrompt, citation: TextCitation
):
    match citation:
        case CitationCharLocation(document_index=document_index):
            await _add_document_citation(consumer, prompt, document_index)

        case CitationPageLocation(
            document_index=document_index, start_page_number=start_page_number
        ):
            extra_url = f"#page={start_page_number}"
            await _add_document_citation(
                consumer, prompt, document_index, extra_url
            )

        # custom document aren't supported yet
        case CitationContentBlockLocation():
            pass
        # web search isn't supported yet
        case CitationsWebSearchResultLocation():
            pass
        case _:
            assert_never(citation)
