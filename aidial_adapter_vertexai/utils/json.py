import json
from typing import Any

import proto
from pydantic import BaseModel
from vertexai.preview.generative_models import Content, Part

from aidial_adapter_vertexai.utils.protobuf import message_to_dict


def json_dumps_short(obj: Any, string_limit: int = 100, *args, **kwargs) -> str:
    return json.dumps(
        _truncate_strings(to_dict(obj), string_limit),
        *args,
        **kwargs,
    )


def to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [to_dict(element) for element in obj]

    if isinstance(obj, BaseModel):
        return to_dict(obj.dict())

    if isinstance(obj, proto.Message):
        return message_to_dict(obj)

    if isinstance(obj, Content):
        return to_dict(obj._raw_content)

    if isinstance(obj, Part):
        return to_dict(obj._raw_part)

    return obj


def _truncate_strings(obj: Any, string_limit: int) -> Any:
    if isinstance(obj, dict):
        return {
            key: _truncate_strings(value, string_limit)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [_truncate_strings(element, string_limit) for element in obj]

    if isinstance(obj, str) and len(obj) > string_limit:
        skip = len(obj) - string_limit
        return (
            obj[: string_limit // 2]
            + f"...({skip} skipped)..."
            + obj[-string_limit // 2 :]
        )

    return obj
