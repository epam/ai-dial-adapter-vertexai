from pathlib import Path

from aidial_adapter_vertexai.utils.resource import Resource

BLUE_PNG_PICTURE = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)

DOG_PICTURE = Resource(
    type="image/jpeg", data=Path("tests/assets/image1.jpg").read_bytes()
)
