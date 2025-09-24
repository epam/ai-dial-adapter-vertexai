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
