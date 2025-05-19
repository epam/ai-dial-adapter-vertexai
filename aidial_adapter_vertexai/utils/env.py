import json
import os
from typing import Dict


def get_env(name: str) -> str:
    if (val := os.getenv(name)) is not None:
        return val

    raise Exception(f"{name} env variable is not set")


def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


def get_str_dict(name: str) -> Dict[str, str]:
    if (val := os.getenv(name)) is None:
        return {}

    try:
        dct = json.loads(val)
        assert isinstance(dct, dict)
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in dct.items()
        )
        return dct
    except Exception:
        raise ValueError(
            f"{name} env variable doesn't contain a valid string to string JSON dictionary"
        )
