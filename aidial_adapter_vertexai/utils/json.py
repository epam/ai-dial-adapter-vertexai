import json
from typing import Any

import proto
from pydantic import BaseModel

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

    # For vertexai.preview.generative_models.[Content|Part]
    msg = getattr(item, "_raw_part", None) or getattr(
        item, "_raw_content", None
    )
    if msg is not None and isinstance(msg, proto.Message):
        return to_dict(message_to_dict(msg))

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
