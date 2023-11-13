from typing import Any, Dict, List, Optional

from typing_extensions import override

from aidial_adapter_vertexai.llm.chat_completion_adapter import (
    ChatCompletionAdapter,
)
from aidial_adapter_vertexai.llm.exceptions import ValidationError
from aidial_adapter_vertexai.llm.vertex_ai_chat import VertexAIMessage
from aidial_adapter_vertexai.universal_api.request import ModelParameters


class BisonChatAdapter(ChatCompletionAdapter):
    @override
    def _create_instance(
        self,
        context: Optional[str],
        messages: List[VertexAIMessage],
    ) -> Dict[str, Any]:
        return {
            "context": context or "",
            "messages": messages,
        }

    @override
    def _create_parameters(
        self,
        params: ModelParameters,
    ) -> Dict[str, Any]:
        # See chat playground: https://console.cloud.google.com/vertex-ai/generative/language/create/chat
        ret: Dict[str, Any] = {}

        if params.max_tokens is not None:
            ret["maxOutputTokens"] = params.max_tokens

        if params.temperature is not None:
            ret["temperature"] = params.temperature

        if params.stop is not None:
            ret["stopSequences"] = (
                [params.stop] if isinstance(params.stop, str) else params.stop
            )

        if params.top_p is not None:
            ret["topP"] = params.top_p

        return ret


class BisonCodeChatAdapter(ChatCompletionAdapter):
    @override
    def _create_instance(
        self,
        context: Optional[str],
        messages: List[VertexAIMessage],
    ) -> Dict[str, Any]:
        if context is not None:
            raise ValidationError("System message is not supported")

        return {
            "messages": messages,
        }

    @override
    def _create_parameters(
        self,
        params: ModelParameters,
    ) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}

        if params.max_tokens is not None:
            ret["maxOutputTokens"] = params.max_tokens

        if params.temperature is not None:
            ret["temperature"] = params.temperature

        if params.stop is not None:
            raise ValidationError(
                "stop sequences are not supported for code chat model"
            )

        if params.top_p is not None:
            raise ValidationError("top_p is not supported for code chat model")

        return ret
