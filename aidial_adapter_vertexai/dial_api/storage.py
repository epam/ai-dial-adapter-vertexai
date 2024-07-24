import hashlib
import io
import mimetypes
import os
from typing import Mapping, Optional, TypedDict
from urllib.parse import urljoin

import aiohttp
from pydantic import BaseModel

from aidial_adapter_vertexai.utils.log_config import app_logger as log


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
    bucket: Optional[Bucket] = None

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
                self.bucket = await response.json()
                log.debug(f"bucket: {self.bucket}")

        return self.bucket

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
        base_url = f"{self.dial_url}/v1/"
        return urljoin(base_url, link)

    async def download_file(self, url: str) -> bytes:
        headers: Mapping[str, str] = {}
        if url.startswith(self.dial_url):
            headers = self.auth_headers

        return await download_file(url, headers)


async def download_file(url: str, headers: Mapping[str, str] = {}) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.read()


def compute_hash_digest(file_content: str) -> str:
    return hashlib.sha256(file_content.encode()).hexdigest()


DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(api_key: str) -> Optional[FileStorage]:
    if DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)
