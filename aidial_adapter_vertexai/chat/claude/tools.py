import json
from typing import assert_never

from anthropic.types import ToolUseBlock

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
