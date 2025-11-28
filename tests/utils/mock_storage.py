import mimetypes
import os
from pathlib import Path
from typing import List

from aidial_adapter_vertexai.dial_api.storage import FileMetadata, FileStorage


class MockFileStorage(FileStorage):
    root_dir: Path
    files: List[Path]

    @classmethod
    def create(cls, root_dir: Path) -> "MockFileStorage":
        root_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            dial_url="http://test-dial-url",
            api_key="test-dial-api-key",
            root_dir=root_dir,
            files=[],
        )

    def _parse_filename(self, name: str) -> int:
        try:
            return int(name.split(".")[0])
        except Exception:
            return 0

    def _get_fresh_file_index(self) -> int:
        if not (files := os.listdir(self.root_dir)):
            return 1

        max_index = max(self._parse_filename(f) for f in files)
        return max_index + 1

    def _get_fresh_filename(self) -> str:
        return f"{self._get_fresh_file_index():0>3}"

    @staticmethod
    def _get_file_extension(content_type: str) -> str:
        return mimetypes.guess_extension(content_type) or ".bin"

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = self._get_file_extension(content_type)
        name = self._get_fresh_filename() + ext

        file = self.root_dir / name
        file.write_bytes(content)
        self.files.append(file)

        return FileMetadata(
            name=name,
            parentPath=os.path.dirname(name),
            bucket="mock-bucket",
            url=f"files/mock-bucket/{name}",
        )

    async def download_file(self, link: str) -> bytes:
        filename = link.removeprefix("files/mock-bucket/")
        return (self.root_dir / filename).read_bytes()

    async def get_human_readable_name(self, link: str) -> str:
        return link.removeprefix("files/mock-bucket/")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        flag = os.getenv("INTEGRATION_TEST_CLEANUP_MOCK_STORAGE", "").lower()
        if flag in ("1", "true"):
            for file in self.files:
                file.unlink(missing_ok=True)

        if not os.listdir(self.root_dir):
            self.root_dir.rmdir()
