import base64
import re
from typing import Optional, Self

from pydantic import BaseModel


class DataURL(BaseModel):
    """
    Encoding of an image as a data URL.
    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
    """

    type: str
    data: str

    @classmethod
    def from_data_url(cls, data_uri: str) -> Optional[Self]:
        pattern = r"^data:([^;]+);base64,(.+)$"
        match = re.match(pattern, data_uri)
        if match is None:
            return None

        mime_type = match.group(1)
        data = match.group(2)

        try:
            base64.b64decode(data, validate=True)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(
            type=mime_type,
            data=data,
        )

    def to_data_url(self) -> str:
        return f"data:{self.type};base64,{self.data}"
