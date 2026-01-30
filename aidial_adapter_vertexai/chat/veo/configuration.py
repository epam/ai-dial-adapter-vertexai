from google.genai.types import (
    GenerateVideosConfigDict,
    ImageDict,
    VideoCompressionQuality,
    VideoGenerationMaskDict,
    VideoGenerationReferenceImageDict,
)
from pydantic import Field

from aidial_adapter_vertexai.utils.pydantic import ExtraAllowModel


class VeoConfig(ExtraAllowModel):
    """The config for generating videos."""

    fps: int | None = Field(
        default=None,
        description="""Frames per second for video generation.""",
    )

    duration_seconds: int | None = Field(
        default=None,
        description="""Duration of the clip for video generation in seconds.""",
    )

    aspect_ratio: str | None = Field(
        default=None,
        description="""The aspect ratio for the generated video. 16:9 (landscape) and 9:16 (portrait) are supported.""",
    )

    resolution: str | None = Field(
        default=None,
        description="""The resolution for the generated video. 720p and 1080p are supported.""",
    )

    person_generation: str | None = Field(
        default=None,
        description="""Whether allow to generate person videos, and restrict to specific ages. Supported values are: dont_allow, allow_adult.""",
    )

    pubsub_topic: str | None = Field(
        default=None,
        description="""The pubsub topic where to publish the video generation progress.""",
    )

    negative_prompt: str | None = Field(
        default=None,
        description="""Explicitly state what should not be included in the generated videos.""",
    )

    enhance_prompt: bool | None = Field(
        default=None,
        description="""Whether to use the prompt rewriting logic.""",
    )

    generate_audio: bool | None = Field(
        default=None,
        description="""Whether to generate audio along with the video.""",
    )

    last_frame: ImageDict | None = Field(
        default=None,
        description="""Image to use as the last frame of generated videos. Only supported for image to video use cases.""",
    )

    reference_images: list[VideoGenerationReferenceImageDict] | None = Field(
        default=None,
        description="""The images to use as the references to generate the videos. If this field is provided, the text prompt field must also be provided. The image, video, or last_frame field are not supported. Each image must be associated with a type. Veo 2 supports up to 3 asset images *or* 1 style image.""",
    )

    mask: VideoGenerationMaskDict | None = Field(
        default=None,
        description="""The mask to use for generating videos.""",
    )

    compression_quality: VideoCompressionQuality | None = Field(
        default=None,
        description="""Compression quality of the generated videos.""",
    )

    def to_config_dict(self, seed: int | None) -> GenerateVideosConfigDict:
        ret: GenerateVideosConfigDict = {
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution,
            "person_generation": self.person_generation,
            "pubsub_topic": self.pubsub_topic,
            "negative_prompt": self.negative_prompt,
            "enhance_prompt": self.enhance_prompt,
            "generate_audio": self.generate_audio,
            "last_frame": self.last_frame,
            "reference_images": self.reference_images,
            "mask": self.mask,
            "compression_quality": self.compression_quality,
            "seed": seed,
            "number_of_videos": 1,
        }

        return ret | self.extra_fields  # type: ignore
