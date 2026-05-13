from pathlib import Path

from aidial_adapter_vertexai.utils.resource import Resource

_CURRENT_DIR = Path(__file__).parent.parent
_ASSETS_DIR = _CURRENT_DIR / "assets"


def _from_assets(type: str, name: str) -> Resource:
    return Resource(
        type=type,
        data=(_ASSETS_DIR / name).read_bytes(),
    )


BLUE_PNG_PICTURE = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)

JPG_RESOURCE = DOG_PICTURE = _from_assets("image/jpeg", "image1.jpg")
DOG_PICTURE_CONTENT = ["dog", "labrador"]

MP3_RESOURCE = _from_assets("audio/mpeg", "audio.mp3")
MP4_RESOURCE = _from_assets("video/mp4", "video.mp4")
MP4_24FPS_RESOURCE = _from_assets("video/mp4", "video_24fps.mp4")
PDF_RESOURCE = _from_assets("application/pdf", "doc.pdf")
OCR_PNG_RESOURCE = _from_assets("image/png", "ocr.png")
BMP_RESOURCE = _from_assets("image/bmp", "image.bmp")
XLSX_RESOURCE = _from_assets(
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "table.xlsx",
)
UNKNOWN_BINARY_RESOURCE = Resource(
    type="application/octet-stream", data=b"1234567890"
)
