from typing import Any, assert_never

from google.genai.types import CountTokensConfigDict as GenAICountTokensConfig
from google.genai.types import (
    GenerateContentConfigDict as GenAIGenerationConfig,
)
from google.genai.types import ImageConfigDict, ThinkingConfigDict
from google.genai.types import Part as GenAIPart
from google.genai.types import ToolDict as GenAITool

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters


def validate_n_parameter(params: ModelParameters) -> None:
    # Currently n>1 is emulated by calling the model n times.
    # So the individual generation requests are expected to have n=1 or unset.
    if params.n is not None and params.n > 1:
        raise ValueError("n is expected to be 1 or unset")


def create_genai_generation_config(
    params: ModelParameters,
    *,
    supports_image_generation: bool,
    tools: ToolsConfig,
    static_tools: StaticToolsConfig,
    system_instruction: list[GenAIPart] | None,
    thinking_config: ThinkingConfigDict | None,
    image_config: ImageConfigDict | None,
) -> GenAIGenerationConfig:
    validate_n_parameter(params)

    config = create_genai_count_tokens_config(
        tools, static_tools, system_instruction
    )

    response_mime_type, response_schema = _get_response_format(params)

    response_modalities = None
    if supports_image_generation:
        response_modalities = ["TEXT", "IMAGE"]

    tool_config = tools.to_gemini_genai_tool_config()

    return GenAIGenerationConfig(
        system_instruction=config.get("system_instruction"),
        tools=config.get("tools"),
        tool_config=tool_config,
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=params.stop,
        top_p=params.top_p,
        candidate_count=params.n,
        seed=params.seed,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        response_modalities=response_modalities,
        thinking_config=thinking_config,
        image_config=image_config,
    )


def create_genai_count_tokens_config(
    tools: ToolsConfig,
    static_tools: StaticToolsConfig,
    system_instruction: list[GenAIPart] | None = None,
) -> GenAICountTokensConfig:
    toolset: GenAITool | None = None
    for tool in tools.to_gemini_genai_tools():
        if toolset is None:
            toolset = tool
            continue

        incoming_declarations = tool.get("function_declarations")
        if incoming_declarations is None:
            continue

        current_declarations = toolset.get("function_declarations")
        if current_declarations is None:
            toolset["function_declarations"] = list(incoming_declarations)
            continue

        current_declarations.extend(incoming_declarations)

    for static_tool in static_tools.to_gemini_genai_tools():
        if toolset is None:
            toolset = static_tool
            continue
        toolset.update(static_tool)

    return GenAICountTokensConfig(
        system_instruction=(
            list(system_instruction) if system_instruction else None
        ),
        tools=[toolset] if toolset is not None else None,
    )


def _get_response_format(
    params: ModelParameters,
) -> tuple[str | None, dict[str, Any] | None]:
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
