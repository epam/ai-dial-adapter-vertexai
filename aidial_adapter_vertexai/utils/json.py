import json
from enum import Enum
from typing import Any

import proto
from pydantic import BaseModel
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    Part,
)

from aidial_adapter_vertexai.utils.protobuf import message_to_dict


def json_dumps_short(obj: Any, string_limit: int = 100, **kwargs) -> str:
    return json.dumps(
        _truncate_strings(to_dict(obj, **kwargs), string_limit),
    )


def json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(to_dict(obj, **kwargs))


def to_dict(obj: Any, **kwargs) -> Any:
    def rec(val):
        return to_dict(val, **kwargs)

    def dict_field(key: str, val: Any) -> Any:
        if key in kwargs.get("excluded_keys", []):
            return "<excluded>"
        return val

    if isinstance(obj, bytes):
        return f"<bytes>({len(obj):_} B)"

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {key: rec(dict_field(key, value)) for key, value in obj.items()}

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, BaseModel):
        return rec(obj.dict())

    if isinstance(obj, proto.Message):
        return rec(message_to_dict(obj))

    if isinstance(obj, GenerationResponse):
        return rec(obj._raw_response)

    if isinstance(obj, GenerationConfig):
        return rec(obj._raw_generation_config)

    if isinstance(obj, Content):
        return rec(obj._raw_content)

    if isinstance(obj, Part):
        return rec(obj._raw_part)

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
            + f"...({skip:_} skipped)..."
            + obj[-string_limit // 2 :]
        )

    return obj
