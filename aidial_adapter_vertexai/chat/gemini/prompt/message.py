import json
from typing import Any

from google.genai.types import Content as GenAIContent
from google.genai.types import Part as GenAIPart

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


def fold_text_into_function_response(content: GenAIContent) -> None:
    """
    Accumulate text content in a closest previous function response:

    parts:
        {function_response=xxx}
        {text=text1}
        {text=text2}
        {function_response=yyy}
        {text=text3}
    ==>
    parts:
        {function_response=xxx\n\ntext1\n\ntext2}
        {function_response=yyy\n\ntext3}
    """

    old_parts = content.parts or []
    new_parts: list[GenAIPart] = []

    i = 0
    while i < len(old_parts):
        new_parts.append(old_parts[i])
        if resp := old_parts[i].function_response:
            i += 1
            texts: list[str] = []
            while i < len(old_parts) and (text := old_parts[i].text):
                texts.append(text)
                i += 1
            resp.response = _append_text(resp.response or {}, texts)
        else:
            i += 1

    content.parts = new_parts


def _append_text(response: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    if not texts:
        return response
    output = response["output"] if response.keys() == {"output"} else response
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    text = "\n\n".join(texts)
    return {"output": f"{output}\n\n{text}"}
