from typing import Callable

from aidial_adapter_vertexai.utils.log_config import app_logger as log


def fail_safe(func: Callable[..., str]) -> Callable[..., str]:
    def _wrapper(*args, **kwargs) -> str:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Error executing {func.__name__}: {str(e)}"
            log.warning(msg)
            return msg

    return _wrapper
