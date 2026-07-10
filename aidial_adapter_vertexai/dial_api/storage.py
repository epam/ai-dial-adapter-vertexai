import hashlib
import io
import mimetypes
import os
from collections.abc import Mapping
from urllib.parse import unquote, urljoin

import aiohttp
from pydantic import BaseModel
from typing_extensions import TypedDict

from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.url import (
    download_public_file,
    has_same_origin,
)


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

        # DIAL Core is the only origin allowed to receive the api-key, and it
        # may legitimately live on a private address. Trust is decided by
        # origin (scheme/host/port), never by string prefix, otherwise a URL
        # like ``http://<dial_url>@169.254.169.254`` would bypass the SSRF
        # check and leak the api-key.
        if not has_same_origin(url, self.dial_url):
            return await download_public_file(url)

        async with (
            aiohttp.ClientSession() as session,
            session.get(url, headers=self.auth_headers) as response,
        ):
            response.raise_for_status()
            return await response.read()

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


def compute_hash_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(api_key: str) -> FileStorage | None:
    if DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)
