from aidial_adapter_vertexai.chat.gemini.processors import (
    get_audio_processor,
)


def test_audio_processor_supports_webm():
    processor = get_audio_processor()

    assert "audio/webm" in processor.mime_types
    assert "webm" in processor.file_exts


def test_audio_processor_rejects_unsupported_mime_type():
    processor = get_audio_processor()

    assert "audio/x-unsupported" not in processor.mime_types
