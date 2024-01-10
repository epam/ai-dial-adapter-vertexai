import json
from typing import Any

from pydantic import BaseModel
from vertexai.preview.generative_models import Content, Part

from aidial_adapter_vertexai.utils.protobuf import message_to_dict


def json_dumps_short(obj: Any, string_limit: int = 100, *args, **kwargs) -> str:
    return json.dumps(
        _truncate_strings(to_dict(obj), string_limit),
        *args,
        **kwargs,
    )


def to_dict(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: to_dict(value) for key, value in item.items()}

    if isinstance(item, list):
        return [to_dict(element) for element in item]

    if isinstance(item, BaseModel):
        return to_dict(item.dict())

    if isinstance(item, Content):
        return message_to_dict(item._raw_content)

    if isinstance(item, Part):
        return message_to_dict(item._raw_part)

    return item


def _truncate_strings(item: Any, string_limit: int) -> Any:
    if isinstance(item, dict):
        return {
            key: _truncate_strings(value, string_limit)
            for key, value in item.items()
        }

    if isinstance(item, list):
        return [_truncate_strings(element, string_limit) for element in item]

    if isinstance(item, str) and len(item) > string_limit:
        skip = len(item) - string_limit
        return (
            item[: string_limit // 2]
            + f"...({skip} skipped)..."
            + item[-string_limit // 2 :]
        )

    return item
