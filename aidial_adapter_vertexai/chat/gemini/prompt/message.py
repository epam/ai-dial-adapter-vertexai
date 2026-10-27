import json
from typing import Any

from google.genai.types import Content as GenAIContent

from aidial_adapter_vertexai.utils.list import MessageMergeStrategy


class GenAIMessageMerger(MessageMergeStrategy[GenAIContent]):
    @staticmethod
    def role(message: GenAIContent) -> str | None:
        return message.role

    @staticmethod
    def merge(a: GenAIContent, b: GenAIContent) -> GenAIContent:
        if a.role != b.role:
            raise ValueError("Cannot merge messages with different roles")
        return GenAIContent(
            role=a.role, parts=(a.parts or []) + (b.parts or [])
        )


def fold_text_into_function_response(content: GenAIContent) -> GenAIContent:
    """
    Move the text parts of a turn that also carries function responses
    into the last of those responses, separated by a blank line.

    Such a turn comes from a user (or tool) message following tool results,
    e.g. a reminder an agent harness appends after a tool call.
    Gemini 3.6 and later reject it with
    "Requests ending with a model turn are not supported",
    and Google's guidance for these models is to append inline instructions
    to the function response text, separated by two newlines.
    Earlier models accept both forms, so the text is folded for every model.
    """
    parts = content.parts or []
    responses = [part for part in parts if part.function_response]
    texts = [part.text for part in parts if part.text]
    if not responses or not texts:
        return content

    last = responses[-1]
    assert last.function_response is not None
    folded = last.model_copy(
        update={
            "function_response": last.function_response.model_copy(
                update={
                    "response": _append_text(
                        last.function_response.response or {},
                        "\n\n".join(texts),
                    )
                }
            )
        }
    )

    return content.model_copy(
        update={
            "parts": [
                folded if part is last else part
                for part in parts
                if part.text is None
            ]
        }
    )


def _append_text(response: dict[str, Any], text: str) -> dict[str, Any]:
    output = response["output"] if response.keys() == {"output"} else response
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    return {"output": f"{output}\n\n{text}"}
