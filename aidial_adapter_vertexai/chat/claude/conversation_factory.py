import base64
import json
from typing import List, Literal, assert_never

from aidial_sdk.chat_completion.request import Role
from aidial_sdk.exceptions import InvalidRequestError
from anthropic.types import (
    Base64PDFSourceParam,
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
    TextBlockParam
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
) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"] | None:
    match mime_type:
        case "image/jpeg" | "image/png" | "image/gif" | "image/webp":
            return mime_type
        case _:
            return None


class ClaudeConversationFactory(
    ConversationFactoryBase[ClaudePart, MessageParam, ClaudeConversation]
):
    def create_multi_modal_part(
        self, data: bytes, mime_type: str
    ) -> ImageBlockParam | DocumentBlockParam:
        text_data = base64.b64encode(data).decode()
        if image_type := _parse_image_type(mime_type):
            source = Source(
                type="base64",
                data=text_data,
                media_type=image_type,
            )
            return ImageBlockParam(type="image", source=source)
        elif mime_type == "application/pdf":
            source = Base64PDFSourceParam(
                type="base64",
                media_type=mime_type,
                data=text_data,
            )
            return DocumentBlockParam(type="document", source=source)
        else:
            raise InvalidRequestError("Unsupported file format")

    def create_text_part(self, text: str) -> ClaudePart:
        return TextBlockParam(type="text", text=text)

    def create_function_call_part(
        self, name: str, args: str, tool_call_id: str
    ) -> ClaudePart:
        return ToolUseBlockParam(
            id=tool_call_id,
            input=json.loads(args),
            name=name,
            type="tool_use",
        )

    def create_function_result_part(
        self, name: str, args: str, tool_call_id: str
    ) -> ClaudePart:
        return ToolResultBlockParam(
            tool_use_id=tool_call_id,
            type="tool_result",
            content=[TextBlockParam(type="text", text=args)],
        )

    def create_content(
        self, role: Role, parts: List[ClaudePart]
    ) -> MessageParam:
        match role:
            case Role.USER | Role.FUNCTION | Role.TOOL:
                return MessageParam(content=parts, role="user")
            case Role.ASSISTANT:
                return MessageParam(content=parts, role="assistant")
            case Role.SYSTEM | Role.DEVELOPER:
                raise InvalidRequestError(
                    "System or developer message is only allowed as the first message"
                )
            case _:
                assert_never(role)

    def create_conversation(
        self,
        system_instruction: List[ClaudePart] | None,
        contents: List[MessageParam],
    ) -> ClaudeConversation:
        return ClaudeConversation.create(
            system=_sanitize_system_instruction(system_instruction),
            messages=contents,
        )


def _sanitize_system_instruction(
    parts: List[ClaudePart] | None,
) -> List[TextBlockParam] | None:
    if parts is None:
        return None

    ret: List[TextBlockParam] = []
    for part in parts:
        if isinstance(part, dict) and part["type"] == "text":
            ret.append(part)
        else:
            raise InvalidRequestError(
                "Only text parts are allowed in the system message"
            )

    return ret
