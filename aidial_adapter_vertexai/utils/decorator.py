from collections.abc import Callable
from typing import ParamSpec

from aidial_adapter_vertexai.utils.log_config import app_logger as log

_P = ParamSpec("_P")


def fail_safe(func: Callable[_P, str]) -> Callable[_P, str]:
    def _wrapper(*args, **kwargs) -> str:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Error executing {func.__name__}: {str(e)}"
            log.warning(msg)
            return msg

    return _wrapper
