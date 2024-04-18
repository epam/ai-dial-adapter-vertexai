import base64
import re
from typing import Optional, Self

from pydantic import BaseModel


class Resource(BaseModel):
    """
    Resource: byte value + MIME type
    """

    mime_type: str
    data: bytes

    @classmethod
    def from_data_url(cls, url: str) -> Optional[Self]:
        """
        See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
        """

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