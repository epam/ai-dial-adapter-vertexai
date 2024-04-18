import base64
import re
from typing import Optional, Self

from pydantic import BaseModel


class DataURL(BaseModel):
    """
    Encoding of an image as a data URL.
    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
    """

    mime_type: str
    data: bytes

    @classmethod
    def from_data_url(cls, url: str) -> Optional[Self]:
        pattern = r"^data:([^;]+);base64,(.+)$"
        match = re.match(pattern, url)
        if match is None:
            return None

        mime_type = match.group(1)
        base64_data = match.group(2)

        try:
            data = base64.b64decode(base64_data, validate=True)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(mime_type=mime_type, data=data)

    def to_data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64_data}"

    @property
    def base64_data(self) -> str:
        return base64.b64encode(self.data).decode("utf-8")
