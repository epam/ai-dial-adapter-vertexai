from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart

from aidial_adapter_vertexai.chat.gemini.state import MessageState

_SIGNATURE = b"signature"


def _content(*parts: GenAIPart) -> GenAIContent:
    return GenAIContent(role="model", parts=list(parts))


def _function_call(name: str) -> GenAIPart:
    return GenAIPart.from_function_call(name=name, args={})


def test_signature_is_attached_to_the_first_function_call():
    state = MessageState()
    state.set_thought_signature(_SIGNATURE)

    content = _content(
        GenAIPart.from_text(text="text"),
        _function_call("first"),
        _function_call("second"),
    )
    state.update_content(content)

    parts = content.parts or []
    assert [part.thought_signature for part in parts] == [
        None,
        _SIGNATURE,
        None,
    ]


def test_signature_is_attached_to_the_last_part_when_no_function_calls():
    state = MessageState()
    state.set_thought_signature(_SIGNATURE)

    content = _content(
        GenAIPart.from_text(text="first"),
        GenAIPart.from_text(text="last"),
    )
    state.update_content(content)

    parts = content.parts or []
    assert [part.thought_signature for part in parts] == [None, _SIGNATURE]


def test_validation_is_disabled_when_signature_is_missing():
    content = _content(
        _function_call("first"),
        _function_call("second"),
    )
    MessageState().update_content(content)

    parts = content.parts or []
    assert [part.thought_signature for part in parts] == [
        b"skip_thought_signature_validator",
        None,
    ]


def test_empty_content_is_left_intact():
    state = MessageState()
    state.set_thought_signature(_SIGNATURE)

    content = _content()
    state.update_content(content)

    assert content.parts == []
