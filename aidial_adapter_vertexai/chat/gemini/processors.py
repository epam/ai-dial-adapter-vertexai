from typing import List

from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessor,
    InitValidator,
    max_count_validator,
    max_pdf_page_count_validator,
    seq_validators,
)

# Gemini capabilities: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts
# Prompt design: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing


# Tokens per image: 258. count_tokens API call takes this into account.
def get_image_processor(
    max_count: int,
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    # NOTE: the validator maintains a state, so we need to create a new instance each time
    return AttachmentProcessor(
        file_types={
            "image/jpeg": ["jpg", "jpeg"],
            "image/png": "png",
        },
        init_validator=seq_validators(
            init_validator, max_count_validator(max_count)
        ),
    )


# The maximum number of hours of audio per prompt is approximately 8.4 hours,
# or up to 1 million tokens (not checked).
def get_audio_processor(
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={"audio/mp3": "mp3", "audio/wav": "wav"},
        init_validator=init_validator,
    )


# The maximum file size for a PDF is 50MB (not checked).
# PDFs are treated as images, so a single page of a PDF is treated as one image.
def get_pdf_processor(
    max_page_count: int,
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={"application/pdf": "pdf"},
        init_validator=init_validator,
        post_validator=max_pdf_page_count_validator(max_page_count),
    )


# For Gemini 1.5 Pro (Preview) only, the audio track is encoded with video frames.
# The audio track is also broken down into 1-second trunks that
# each accounts for 32 tokens.
# The video frame and audio tokens are interleaved together with their timestamps. # The timestamps are represented as 7 tokens.
# Maximum video length when it includes audio is approximately 50 minutes (not checked).
# The maximum video length for video without audio is 1 hour (not checked).


# Audio in the video is ignored.
# Videos are sampled at 1fps. Each video frame accounts for 258 tokens.
# The video is automatically truncated to the first two minutes.
def get_video_processor(
    max_count: int,
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={
            "video/mp4": "mp4",
            "video/mov": "mov",
            "video/mpeg": "mpeg",
            "video/mpg": "mpg",
            "video/avi": "avi",
            "video/wmv": "wmv",
            "video/mpegps": "mpegps",
            "video/flv": "flv",
        },
        init_validator=seq_validators(
            init_validator, max_count_validator(max_count)
        ),
    )


def get_file_exts(processors: List[AttachmentProcessor]) -> List[str]:
    return [ext for proc in processors for ext in proc.file_exts]
