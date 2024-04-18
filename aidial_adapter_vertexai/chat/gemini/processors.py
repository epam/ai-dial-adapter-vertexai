from typing import List

from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessor,
    InitValidator,
    max_count_validator,
    max_pdf_page_count_validator,
    seq_validators,
)

# Gemini capabilities: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts
# More on it: https://ai.google.dev/gemini-api/docs/prompting_with_media
# Prompt design: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing

# Text/Code processing:
# 1.0: max_total_tokens: 16384, max_completion_tokens: 2048
# 1.5: max_total_tokens ~: 1M, max_completion_tokens: not specified


# Image processing:
# 1.0:
#  * max number of images: 16
#  * Tokens per image: 258. count_tokens API call takes this into account.
# 1.5: max number of images: 3000
def get_image_processor(
    max_count: int,
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    # NOTE: the validator maintains a state, so we need to create a new instance each time
    return AttachmentProcessor(
        file_types={
            "image/jpeg": ["jpg", "jpeg"],
            "image/png": "png",
            "image/webp": "webp",
            "image/heic": "heic",
            "image/heif": "heif",
        },
        init_validator=seq_validators(
            init_validator, max_count_validator(max_count)
        ),
    )


# Audio processing
# 1.0: not supported
# 1.5: the maximum number of hours of audio per prompt is approximately 8.4 hours,
#      or up to 1 million tokens (not checked).
def get_audio_processor(
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/aiff": "aiff",
            "audio/acc": "acc",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
        },
        init_validator=init_validator,
    )


# PDF processing
# 1.0: max number of PDF pages: 16
# 1.5: max number of PDF pages: 3000
# The maximum file size for a PDF is 50MB (not checked).
# PDF pages are treated as individual images.
def get_pdf_processor(
    max_page_count: int,
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={"application/pdf": "pdf"},
        init_validator=init_validator,
        post_validator=max_pdf_page_count_validator(max_page_count),
    )


# Video processing
# 1.0:
#   * Maximum video length is 2 minutes (not checked)
#   * The maximum number of videos: 1
#   * Audio in the video is ignored.
#   * Videos are sampled at 1fps. Each video frame accounts for 258 tokens.
# 1.5:
#   * the audio track is encoded with video frames.
#   * The audio track is also broken down into 1-second trunks that each accounts for 32 tokens.
#   * The video frame and audio tokens are interleaved together with their timestamps. #   * The timestamps are represented as 7 tokens.
#   * Maximum video length when it includes audio is approximately 50 minutes (not checked)
#   * The maximum video length for video without audio is 1 hour (not checked)
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


def get_mime_types(processors: List[AttachmentProcessor]) -> List[str]:
    return [mime for proc in processors for mime in proc.mime_types]
