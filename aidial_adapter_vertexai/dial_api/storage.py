import hashlib
import io
import mimetypes
import os
from collections.abc import Mapping
from urllib.parse import unquote, urljoin, urlsplit

import aiohttp
from pydantic import BaseModel
from typing_extensions import TypedDict

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api.ssrf import validate_public_url
from aidial_adapter_vertexai.utils.log_config import app_logger as log

# Redirects have to be followed manually so that every hop can be validated
# against SSRF, otherwise a public URL could redirect to an internal address.
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    return (
        scheme,
        (parsed.hostname or "").lower(),
        (parsed.port or _DEFAULT_PORTS.get(scheme)),
    )


def _same_origin(a: str, b: str) -> bool:
    return _origin(a) == _origin(b)


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str


class Bucket(TypedDict):
    bucket: str
    appdata: str


class FileStorage(BaseModel):
    dial_url: str
    api_key: str
    bucket: Bucket | None = None

    @property
    def auth_headers(self) -> Mapping[str, str]:
        return {"api-key": self.api_key}

    async def _get_bucket(self, session: aiohttp.ClientSession) -> Bucket:
        if self.bucket is None:
            async with session.get(
                f"{self.dial_url}/v1/bucket",
                headers=self.auth_headers,
            ) as response:
                response.raise_for_status()
                self.bucket = bucket = await response.json()
                log.debug(f"bucket: {self.bucket}")
                return bucket

        return self.bucket

    async def _get_user_bucket(self, session: aiohttp.ClientSession) -> str:
        bucket = await self._get_bucket(session)
        appdata = bucket.get("appdata")
        if appdata is None:
            raise ValueError(
                "Can't retrieve user bucket because appdata isn't available"
            )
        return appdata.split("/", 1)[0]

    @staticmethod
    def _to_form_data(
        filename: str, content_type: str, content: bytes
    ) -> aiohttp.FormData:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(content),
            filename=filename,
            content_type=content_type,
        )
        return data

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            bucket = await self._get_bucket(session)

            appdata = bucket["appdata"]
            ext = mimetypes.guess_extension(content_type) or ""
            url = f"{self.dial_url}/v1/files/{appdata}/{filename}{ext}"

            data = FileStorage._to_form_data(filename, content_type, content)

            async with session.put(
                url=url,
                data=data,
                headers=self.auth_headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                log.debug(f"Uploaded file: url={url}, metadata={meta}")
                return meta

    def attachment_link_to_url(self, link: str) -> str:
        return urljoin(f"{self.dial_url}/v1/", link)

    def _url_to_attachment_link(self, url: str) -> str:
        return url.removeprefix(f"{self.dial_url}/v1/")

    async def download_file(self, link: str) -> bytes:
        url = self.attachment_link_to_url(link)
        # ``trusted_origin`` marks DIAL Core as the only host that may be
        # reached without SSRF validation and that may receive the api-key.
        # It is compared by origin (scheme/host/port), never by string
        # prefix, otherwise URLs like ``http://<dial_url>@169.254.169.254``
        # would bypass the check and leak the api-key.
        return await download_file(
            url, self.auth_headers, trusted_origin=self.dial_url
        )

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        link = self._url_to_attachment_link(url)

        link = link.removeprefix("files/")

        if link.startswith("public/"):
            bucket = "public"
        else:
            async with aiohttp.ClientSession() as session:
                bucket = await self._get_user_bucket(session)

        link = link.removeprefix(f"{bucket}/")
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)


async def download_file(
    url: str,
    headers: Mapping[str, str] = {},
    *,
    trusted_origin: str | None = None,
) -> bytes:
    async with aiohttp.ClientSession() as session:
        for _ in range(_MAX_REDIRECTS + 1):
            # A hop is trusted only when it targets exactly the DIAL Core
            # origin. Every other hop (including redirects that leave that
            # origin) must be validated and must not receive the api-key.
            trusted = trusted_origin is not None and _same_origin(
                url, trusted_origin
            )

            if not trusted:
                await validate_public_url(url)

            async with session.get(
                url,
                headers=headers if trusted else {},
                allow_redirects=False,
            ) as response:
                if response.status in _REDIRECT_STATUSES and (
                    location := response.headers.get("Location")
                ):
                    # Re-validate the redirect target on the next iteration.
                    url = urljoin(url, location)
                    continue

                response.raise_for_status()
                return await response.read()

    raise ValidationError("The file URL has too many redirects")


def compute_hash_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(api_key: str) -> FileStorage | None:
    if DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)
