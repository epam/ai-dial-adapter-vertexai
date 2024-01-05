import mimetypes
from typing import Optional

from aidial_sdk.chat_completion import Attachment

from aidial_adapter_vertexai.utils.image_data_url import ImageDataURL
from aidial_adapter_vertexai.utils.log_config import app_logger as logger
from aidial_adapter_vertexai.utils.storage import (
    FileStorage,
    download_file_as_base64,
)

# Officially supported image types by Gemini Pro Vision
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png"]

# NOTE: Gemini also supports video: "mkv", "mov", "mp4", "webm"

USAGE = f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them.

Supported image types: {', '.join(SUPPORTED_FILE_EXTS)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()


def guess_attachment_type(attachment: Attachment) -> Optional[str]:
    type = attachment.type
    if type is None:
        return None

    if "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        url = attachment.url
        if url is not None:
            url_type = mimetypes.guess_type(url)[0]
            if url_type is not None:
                return url_type
        return None

    return type


async def download_image(
    file_storage: Optional[FileStorage], attachment: Attachment
) -> ImageDataURL | str:
    try:
        type = guess_attachment_type(attachment)
        if type is None:
            return "Can't derive media type of the attachment"
        elif type not in SUPPORTED_IMAGE_TYPES:
            return f"The attachment isn't one of the supported types: {type}"

        if attachment.data is not None:
            return ImageDataURL(type=type, data=attachment.data)

        if attachment.url is not None:
            attachment_link: str = attachment.url

            image_url = ImageDataURL.from_data_url(attachment_link)
            if image_url is not None:
                if image_url.type in SUPPORTED_IMAGE_TYPES:
                    return image_url
                else:
                    return (
                        "The image attachment isn't one of the supported types"
                    )

            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return ImageDataURL(type=type, data=data)

        return "Invalid attachment"

    except Exception as e:
        logger.debug(f"Failed to download image: {e}")
        return "Failed to download image"
