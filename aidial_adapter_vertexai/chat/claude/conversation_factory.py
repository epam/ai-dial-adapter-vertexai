import base64
import json
from typing import List, Literal, assert_never

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

from aidial_adapter_vertexai.chat.attachment_processor import FileTypes
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

SUPPORTED_IMAGE_TYPES: FileTypes = {
    "image/jpeg": ["jpg", "jpeg"],
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}


def _parse_image_type(
    mime_type: str,
) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"]:
    match mime_type:
        case "image/jpeg" | "image/png" | "image/gif" | "image/webp":
            return mime_type
        case _:
            raise InvalidRequestError("Unsupported image format")


class ClaudeConversationFactory(
    ConversationFactoryBase[ClaudePart, MessageParam, ClaudeConversation]
):
    def create_multi_modal_part(
        self, data: bytes, mime_type: str
    ) -> ImageBlockParam:
        source = Source(
            type="base64",
            data=base64.b64encode(data).decode(),
            media_type=_parse_image_type(mime_type),
        )
        return ImageBlockParam(type="image", source=source)

    def create_text_part(self, text: str) -> ClaudePart:
        return TextBlockParam(type="text", text=text)

    def create_function_call_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> ClaudePart:
        return ToolUseBlockParam(
            id=tool_call_id or "123",  # fixme
            input=json.loads(args),
            name=name,
            type="tool_use",
        )

    def create_function_result_part(
        self, name: str, args: str, *, tool_call_id: str | None = None
    ) -> ClaudePart:
        return ToolResultBlockParam(
            tool_use_id=tool_call_id or "123",  # fixme
            type="tool_result",
            content=[{"type": "text", "text": args}],
        )

    def create_content(
        self, role: Role, parts: List[ClaudePart]
    ) -> MessageParam:
        match role:
            case Role.USER | Role.FUNCTION | Role.TOOL:
                return MessageParam(content=parts, role="user")  # type: ignore
            case Role.ASSISTANT:
                return MessageParam(content=parts, role="assistant")  # type: ignore
            case Role.SYSTEM:
                raise InvalidRequestError(
                    "System message is only allowed as the first message"
                )
            case _:
                assert_never(role)

    def create_conversation(
        self,
        system_instruction: List[ClaudePart] | None,
        contents: List[MessageParam],
    ) -> ClaudeConversation:
        return ClaudeConversation.create(
            system_instruction,  # type: ignore # FIXME
            contents,
        )
