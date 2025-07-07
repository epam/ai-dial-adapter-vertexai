from google.genai.types import (
    GenerateImagesConfigDict,
    PersonGeneration,
    SafetyFilterLevel,
)
from pydantic.v1 import Field

from aidial_adapter_vertexai.utils.pydantic import ExtraAllowModel


class ImagenConfig(ExtraAllowModel):
    """The config for generating an images."""

    negative_prompt: str | None = Field(
        default=None,
        description="Description of what to discourage in the generated images.",
    )

    aspect_ratio: str | None = Field(
        default=None, description="Aspect ratio of the generated images."
    )

    guidance_scale: float | None = Field(
        default=None,
        description="Controls how much the model adheres to the text prompt. Large values increase output and prompt alignment, but may compromise image quality.",
    )

    safety_filter_level: SafetyFilterLevel | None = Field(
        default=None, description="Filter level for safety filtering."
    )

    person_generation: PersonGeneration | None = Field(
        default=None,
        description="Allows generation of people by the model.",
    )

    output_mime_type: str | None = Field(
        default=None,
        description="MIME type of the generated image.",
    )

    output_compression_quality: int | None = Field(
        default=None,
        description="Compression quality of the generated image (for `image/jpeg` only).",
    )

    add_watermark: bool | None = Field(
        default=None,
        description="Whether to add a watermark to the generated images.",
    )

    enhance_prompt: bool | None = Field(
        default=None,
        description="Whether to use the prompt rewriting logic.",
    )

    def to_config_dict(self, seed: int | None) -> GenerateImagesConfigDict:
        ret: GenerateImagesConfigDict = {
            "seed": seed,
            "negative_prompt": self.negative_prompt,
            "add_watermark": self.add_watermark,
            "aspect_ratio": self.aspect_ratio,
            "enhance_prompt": self.enhance_prompt,
            "guidance_scale": self.guidance_scale,
            "output_compression_quality": self.output_compression_quality,
            "output_mime_type": self.output_mime_type,
            "safety_filter_level": self.safety_filter_level,
            "person_generation": self.person_generation,
            "include_rai_reason": True,
            "include_safety_attributes": True,
            "number_of_images": 1,
        }

        return ret | self.extra_fields  # type: ignore
