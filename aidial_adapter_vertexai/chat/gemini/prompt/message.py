from google.genai.types import Content as GenAIContent
from vertexai.preview.generative_models import Content

from aidial_adapter_vertexai.utils.list import MessageMergeStrategy


class LegacyMessageMerger(MessageMergeStrategy[Content]):
    @staticmethod
    def role(message: Content) -> str:
        return message.role

    @staticmethod
    def merge(a: Content, b: Content) -> Content:
        if a.role != b.role:
            raise ValueError("Cannot merge messages with different roles")
        return Content(role=a.role, parts=a.parts + b.parts)


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
