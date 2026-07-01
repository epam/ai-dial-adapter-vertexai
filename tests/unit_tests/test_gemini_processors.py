from aidial_adapter_vertexai.chat.gemini.processors import (
    get_audio_processor,
)


def test_audio_processor_supports_webm():
    processor = get_audio_processor()

    assert "audio/webm" in processor.mime_types
    assert "webm" in processor.file_exts


def test_audio_processor_supports_documented_mime_types():
    processor = get_audio_processor()

    for mime_type in [
        "audio/m4a",
        "audio/mpga",
        "audio/mp4",
        "audio/pcm",
    ]:
        assert mime_type in processor.mime_types


def test_audio_processor_rejects_unsupported_mime_type():
    processor = get_audio_processor()

    assert "audio/x-unsupported" not in processor.mime_types
