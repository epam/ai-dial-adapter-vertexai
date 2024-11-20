from aidial_adapter_vertexai.chat.gemini.processor import (
    AttachmentProcessor,
    InitValidator,
    max_count_validator,
    max_pdf_page_count_validator,
    seq_validators,
)

# Gemini capabilities: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts
# Using File API from google-generativeai lib: https://ai.google.dev/gemini-api/docs/prompting_with_media (not useful for us, because it requires Google API key)
# Which combinations of parts are supported: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#request_body
# Prompt design: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing

# Text/Code processing:
# 1.0: max_total_tokens: 16384, max_completion_tokens: 2048
# 1.5: max_total_tokens ~: 1M, max_completion_tokens: not specified


# Plain text file processing:
#   Counts as text processing.
def get_plain_text_processor(
    init_validator: InitValidator | None = None,
) -> AttachmentProcessor:
    return AttachmentProcessor(
        file_types={
            "text/plain": "txt",
            "text/html": ["html", "htm"],
            "text/css": "css",
            "text/javascript": "js",
            "application/x-javascript": "js",
            "text/x-typescript": "ts",
            "application/x-typescript": "ts",
            "text/csv": "csv",
            "text/markdown": "md",
            "text/x-python": "py",
            "application/x-python-code": "py",
            "application/json": "json",
            "text/xml": "xml",
            "application/rtf": "rtf",
            "text/rtf": "rtf",
        },
        init_validator=init_validator,
    )


# Image processing:
# 1.0:
#  * max number of images: 16
#  * Tokens per image: 258. count_tokens API call takes this into account.
# 1.5: max number of images: 3000
def get_image_processor(
    max_count: int, init_validator: InitValidator | None = None
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
# 1.5: max number of PDF pages: 300
# The maximum file size for a PDF is 50MB (not checked).
# PDF pages are treated as individual images.
def get_pdf_processor(
    max_page_count: int, init_validator: InitValidator | None = None
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
    max_count: int, init_validator: InitValidator | None = None
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
            "video/x-flv": "flv",
            "video/webm": "webm",
            "video/3gpp": "3gpp",
        },
        init_validator=seq_validators(
            init_validator, max_count_validator(max_count)
        ),
    )
