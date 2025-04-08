from typing import Any, Dict, List, Tuple

from aidial_sdk.exceptions import InvalidRequestError
from google.genai.types import (
    GenerateContentConfigDict as GenAIGenerationConfig,
)
from google.genai.types import Part as GenAIPart
from typing_extensions import assert_never
from vertexai.preview.generative_models import GenerationConfig

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters


def validate_n_parameter(params: ModelParameters) -> None:
    # Currently n>1 is emulated by calling the model n times.
    # So the individual generation requests are expected to have n=1 or unset.
    if params.n is not None and params.n > 1:
        raise ValueError("n is expected to be 1 or unset")


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    validate_n_parameter(params)

    response_mime_type, response_schema = _get_response_format(params)

    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=params.stop,
        top_p=params.top_p,
        candidate_count=params.n,
        seed=params.seed,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )


def create_genai_generation_config(
    params: ModelParameters,
    is_gemini_1_5: bool,
    tools: ToolsConfig,
    static_tools: StaticToolsConfig,
    system_instruction: List[GenAIPart] | None = None,
) -> GenAIGenerationConfig:
    validate_n_parameter(params)
    genai_tools = None
    if not static_tools.is_empty() and not tools.is_empty():
        raise InvalidRequestError(
            "Using both 'tools' and 'static_tools' simultaneously is not supported."
        )
    elif not tools.is_empty():
        genai_tools = tools.to_gemini_genai_tools()
    elif not static_tools.is_empty():
        genai_tools = static_tools.to_gemini_genai_tools(is_gemini_1_5)

    response_mime_type, response_schema = _get_response_format(params)

    return GenAIGenerationConfig(
        system_instruction=(
            list(system_instruction) if system_instruction else None
        ),
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=params.stop,
        top_p=params.top_p,
        candidate_count=params.n,
        tools=list(genai_tools) if genai_tools else None,
        seed=params.seed,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )


def _get_response_format(
    params: ModelParameters,
) -> Tuple[str | None, Dict[str, Any] | None]:
    if resp_format := params.response_format:
        match resp_format.type:
            case "text":
                return ("text/plain", None)
            case "json_object":
                return ("application/json", None)
            case "json_schema":
                return ("application/json", resp_format.json_schema.schema_)
            case _:
                assert_never(resp_format.type)
    return (None, None)
