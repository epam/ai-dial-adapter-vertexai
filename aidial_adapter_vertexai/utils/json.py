"""
Utilities for pretty-printing JSON in debug logs.
These functions are useful for dumping large data structures,
with options to trim long strings and lists to specified limits.
"""

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, TypeAlias

import proto
from openai import Omit
from pydantic import BaseModel
from pydantic import BaseModel as BaseModelV1

from aidial_adapter_vertexai.utils.decorator import fail_safe
from aidial_adapter_vertexai.utils.protobuf import message_to_dict

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_RECURSIVE_JSON_NOT_SUPPORTED = "Recursive JSON schemas aren't supported"


@fail_safe
def json_dumps_short(
    obj: Any, *, string_limit: int = 100, list_len_limit: int = 10, **kwargs
) -> str:
    def default(obj) -> str:
        return _truncate_strings(str(obj), string_limit)

    return json.dumps(
        _truncate_lists(
            _truncate_strings(_to_dict(obj, **kwargs), string_limit),
            list_len_limit,
        ),
        default=default,
    )


@fail_safe
def json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(_to_dict(obj, **kwargs), default=str)


def _to_dict(obj: Any, **kwargs) -> Any:
    def rec(val):
        return _to_dict(val, **kwargs)

    def dict_field(key: str, val: Any) -> Any:
        if key in kwargs.get("excluded_keys", []):
            return "<excluded>"
        return val

    if isinstance(obj, bytes):
        return f"<bytes>({len(obj):_} B)"

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        filters = []
        if kwargs.get("exclude_none"):
            filters.append(lambda x: x is not None)
        if kwargs.get("exclude_empty_dict"):
            filters.append(lambda x: x != {})

        ret = {}
        for k, v in obj.items():
            v = rec(dict_field(k, v))
            if all(f(v) for f in filters):
                ret[k] = v
        return ret

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    if isinstance(obj, BaseModelV1):
        return rec(obj.model_dump())

    if isinstance(obj, BaseModel):
        return rec(obj.model_dump())

    if isinstance(obj, proto.Message):
        return rec(message_to_dict(obj))

    if isinstance(obj, Omit):
        return "omit"

    if hasattr(obj, "to_dict"):
        return rec(obj.to_dict())

    if is_dataclass(type(obj)):
        return rec(asdict(obj))

    return obj


def _truncate_strings(obj: Any, limit: int) -> Any:
    def rec(val):
        return _truncate_strings(val, limit)

    if isinstance(obj, dict):
        return {key: rec(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    if isinstance(obj, str) and len(obj) > limit:
        skip = len(obj) - limit
        return (
            obj[: limit // 2] + f"...({skip:_} skipped)..." + obj[-limit // 2 :]
        )

    return obj


def _truncate_lists(obj: Any, limit: int) -> Any:
    def rec(val):
        return _truncate_lists(val, limit)

    if isinstance(obj, dict):
        return {key: rec(value) for key, value in obj.items()}

    if isinstance(obj, list):
        if len(obj) > limit:
            skip = len(obj) - limit
            obj = (
                obj[: limit // 2]
                + [f"...({skip:_} skipped)..."]
                + obj[-limit // 2 :]
            )
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    return obj


def to_json_object_or_string(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    try:
        return json.loads(value)
    except ValueError:
        return value


def inline_local_json_refs(schema: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, JsonValue] = deepcopy(schema)
    defs_raw = root.get("$defs")
    defs: dict[str, JsonValue] = defs_raw if isinstance(defs_raw, dict) else {}

    def _inline(node: JsonValue, ref_stack: tuple[str, ...] = ()) -> JsonValue:
        if isinstance(node, list):
            return [_inline(item, ref_stack) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/", 2)[-1]
                if key in ref_stack:
                    raise ValueError(_RECURSIVE_JSON_NOT_SUPPORTED)
                target = defs.get(key)
                if isinstance(target, dict):
                    resolved_target = _inline(
                        deepcopy(target), ref_stack + (key,)
                    )
                    if not isinstance(resolved_target, dict):
                        raise ValueError(_RECURSIVE_JSON_NOT_SUPPORTED)
                    # Keep sibling constraints while replacing the ref.
                    siblings = {
                        k: _inline(v, ref_stack)
                        for k, v in node.items()
                        if k != "$ref"
                    }
                    return {**resolved_target, **siblings}
        return {k: _inline(v, ref_stack) for k, v in node.items()}

    normalized = _inline(root)
    if not isinstance(normalized, dict):
        raise ValueError("JSON schema root must be an object")
    normalized.pop("$defs", None)
    return normalized
