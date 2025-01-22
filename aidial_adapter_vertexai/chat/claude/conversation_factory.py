import json
from typing import List, Literal

from aidial_sdk.chat_completion.request import Role
from aidial_sdk.exceptions import InvalidRequestError
from anthropic.types import (
    ContentBlock,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from aidial_adapter_vertexai.chat.claude.prompt.base import ClaudeConversation
from aidial_adapter_vertexai.chat.conversation.factory import (
    ConversationFactoryBase,
)

ClaudePart = (
    str
    | TextBlockParam
    | ImageBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | DocumentBlockParam
    | ContentBlock
)


class ClaudeConversationFactory(
    ConversationFactoryBase[ClaudePart, MessageParam, ClaudeConversation]
):
    def create_multi_modal_part(
        self, data: bytes, mime_type: str
    ) -> ImageBlockParam:
        if mime_type not in {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }:
            raise InvalidRequestError("Unsupported image format")

        media_type: Literal[
            "image/jpeg", "image/png", "image/gif", "image/webp"
        ] = mime_type  # type: ignore

        source: Source = {
            "type": "base64",
            "data": data.decode("utf-8"),
            "media_type": media_type,
        }
        return {"type": "image", "source": source}

    def create_text_part(self, text: str) -> ClaudePart:
        return text

    def create_function_call_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> ClaudePart:
        return {
            "id": tool_call_id or "123",  # fixme
            "input": json.loads(args),
            "name": name,
            "type": "tool_use",
        }

    def create_function_result_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> ClaudePart:
        return {
            "tool_use_id": tool_call_id or "123",  # fixme
            "type": "tool_result",
            "content": [{"type": "text", "text": args}],
        }

    def create_content(
        self, role: Role, parts: List[ClaudePart]
    ) -> MessageParam:
        # FIXME:
        if len(parts) == 1 and isinstance(parts[0], str):
            content = parts[0]
        else:
            content = parts

        if role == Role.USER:
            return {"content": content, "role": "user"}  # type: ignore
        elif role == Role.ASSISTANT:
            return {"content": content, "role": "assistant"}  # type: ignore
        else:
            raise ValueError(f"Unexpected role: {role.value}")

    def create_conversation(
        self,
        system_instruction: List[ClaudePart] | None,
        contents: List[MessageParam],
    ) -> ClaudeConversation:
        return ClaudeConversation(
            system=system_instruction,  # type: ignore
            messages=contents,
        )
