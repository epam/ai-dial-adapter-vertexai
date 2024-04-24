from abc import ABC
from typing import List

from pydantic import BaseModel
from vertexai.preview.generative_models import ChatSession, Content, Part


class GeminiPrompt(BaseModel, ABC):
    history: List[Content]
    prompt: List[Part]

    class Config:
        arbitrary_types_allowed = True

    @property
    def contents(self) -> List[Content]:
        return self.history + [
            Content(role=ChatSession._USER_ROLE, parts=self.prompt)
        ]
