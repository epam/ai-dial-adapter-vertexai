import json

from google.genai.types import Content as GenAIContent
from google.genai.types import FunctionResponse as GenAIFunctionResponse
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
    Accumulate the text parts in the closest preceding function response,
    or in the first one when no response precedes them:

    parts:
        {text=text1}
        {function_response=xxx}
        {text=text2}
        {text=text3}
        {function_response=yyy}
        {text=text4}
    ==>
    parts:
        {function_response=xxx\n\ntext1\n\ntext2\n\ntext3}
        {function_response=yyy\n\ntext4}

    Gemini 3.6+ rejects a turn that mixes function responses with text.
    """

    target: GenAIFunctionResponse | None = None
    texts: list[str] = []

    new_parts: list[GenAIPart] = []

    for part in content.parts or []:
        if part.text is not None:
            texts.append(part.text)
            continue

        if resp := part.function_response:
            _append_texts(target or resp, texts)
            target, texts = resp, []

        new_parts.append(part)

    if target:
        _append_texts(target, texts)
        content.parts = new_parts


def _append_texts(
    function_response: GenAIFunctionResponse, texts: list[str]
) -> None:
    if not texts:
        return

    response = function_response.response or {}
    output = response["output"] if response.keys() == {"output"} else response
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)

    function_response.response = {"output": "\n\n".join([output, *texts])}
