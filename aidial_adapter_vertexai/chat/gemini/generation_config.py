from vertexai.preview.generative_models import GenerationConfig

from aidial_adapter_vertexai.dial_api.request import ModelParameters


def validate_n_parameter(params: ModelParameters) -> None:
    # Currently n>1 is emulated by calling the model n times.
    # So the individual generation requests are expected to have n=1 or unset.
    if params.n is not None and params.n > 1:
        raise ValueError("n is expected to be 1 or unset")


def create_generation_config(params: ModelParameters) -> GenerationConfig:
    validate_n_parameter(params)
    return GenerationConfig(
        max_output_tokens=params.max_tokens,
        temperature=params.temperature,
        stop_sequences=params.stop,
        top_p=params.top_p,
        candidate_count=params.n,
    )
