import base64
import mimetypes
import re
from pathlib import Path
from typing import Optional, Self

from aidial_sdk.chat_completion import Attachment
from pydantic import BaseModel
from vertexai.preview.generative_models import Part


class Resource(BaseModel):
    """
    Resource: byte value + MIME type
    """

    mime_type: str
    data: bytes

    @classmethod
    def from_path(cls, path: Path) -> Self:
        mime_type = mimetypes.guess_type(path)[0]
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for {path}")
        return cls(mime_type=mime_type, data=path.read_bytes())

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

    def get_base64_data(self) -> str:
        return base64.b64encode(self.data).decode()

    def to_part(self) -> Part:
        return Part.from_data(data=self.data, mime_type=self.mime_type)

    def to_attachment(self) -> Attachment:
        return Attachment(
            type=self.mime_type,
            data=self.get_base64_data(),
        )
