import base64
import re
from typing import Optional

from pydantic import BaseModel


class Resource(BaseModel):
    type: str
    data: str

    @classmethod
    def from_data_url(cls, data_url: str) -> Optional["Resource"]:
        """
        Parsing a resource encoded as a data URL.
        See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
        """

        type = cls.parse_data_url_content_type(data_url)
        if type is None:
            return None

        data = data_url.removeprefix(cls._to_data_url_prefix(type))

        try:
            base64.b64decode(data)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(type=type, data=data)

    @property
    def data_bytes(self) -> bytes:
        return self.data.encode()

    def to_data_url(self) -> str:
        return f"{self._to_data_url_prefix(self.type)}{self.data}"

    @staticmethod
    def parse_data_url_content_type(data_url: str) -> Optional[str]:
        pattern = r"^data:([^;]+);base64,"
        match = re.match(pattern, data_url)
        return None if match is None else match.group(1)

    @staticmethod
    def _to_data_url_prefix(content_type: str) -> str:
        return f"data:{content_type};base64,"

    def __str__(self) -> str:
        return self.to_data_url()[:100] + "..."
