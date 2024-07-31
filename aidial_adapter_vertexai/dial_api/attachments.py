import base64
import mimetypes
from typing import List

from aidial_sdk.chat_completion import Attachment

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.dial_api.storage import FileStorage, download_file
from aidial_adapter_vertexai.utils.resource import Resource


def derive_attachment_mime_type(attachment: Attachment) -> str | None:
    type = attachment.type
    url = attachment.url

    if url is not None:
        if type is None:
            # No type is provided. Trying to guess the type from the Data URL
            resource = Resource.from_data_url(url)
            if resource is not None:
                return resource.mime_type

        if type is None or "octet-stream" in type:
            # It's an arbitrary binary file. Trying to guess the type from the URL
            mime_type = mimetypes.guess_type(url)[0]
            if mime_type is not None:
                return mime_type

    return type


async def download_attachment(
    file_storage: FileStorage | None, attachment: Attachment
) -> bytes:
    if attachment.data is not None:
        try:
            return base64.b64decode(attachment.data, validate=True)
        except Exception:
            raise ValidationError(
                "Data field of an attachment is expected to be base64-encoded"
            )

    if attachment.url is not None:
        attachment_link: str = attachment.url

        resource = Resource.from_data_url(attachment_link)
        if resource is not None:
            return resource.data

        if file_storage is not None:
            url = file_storage.attachment_link_to_url(attachment_link)
            return await file_storage.download_file(url)

        return await download_file(attachment_link)

    raise ValueError("Invalid attachment: neither url nor data is provided")


async def download_with_content_type(
    supported_content_types: List[str],
    file_storage: FileStorage | None,
    attachment: Attachment,
) -> bytes:
    content_type = derive_attachment_mime_type(attachment)

    if content_type is None:
        raise ValidationError("The attachment type is not provided")

    if content_type not in supported_content_types:
        raise UserError(
            f"Unsupported content type: {content_type}. Supported types: {', '.join(supported_content_types)}."
        )

    return await download_attachment(file_storage, attachment)
