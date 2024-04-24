from functools import wraps

from aidial_sdk import HTTPException as DialException
from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
)
from google.auth.exceptions import GoogleAuthError

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.utils.log_config import app_logger as log


def to_dial_exception(e: Exception) -> DialException:
    if isinstance(e, GoogleAuthError):
        return DialException(
            status_code=401,
            type="invalid_request_error",
            message=f"Invalid Authentication: {str(e)}",
            code="invalid_api_key",
            param=None,
        )

    if isinstance(e, PermissionDenied):
        return DialException(
            status_code=403,
            type="invalid_request_error",
            message=f"Permission denied: {str(e)}",
            code="permission_denied",
            param=None,
        )

    if isinstance(e, InvalidArgument):
        # Imagen content filtering message
        content_filter_msg = (
            "The response is blocked, as it may violate our policies."
        )
        if content_filter_msg in str(e):
            return DialException(
                status_code=400,
                type="invalid_request_error",
                message=content_filter_msg,
                code="content_filter",
                param="prompt",
            )

        return DialException(
            status_code=400,
            type="invalid_request_error",
            message=f"Invalid argument: {str(e)}",
            code="invalid_argument",
            param=None,
        )

    if isinstance(e, GoogleAPICallError):
        return DialException(
            status_code=e.code or 500,
            type="invalid_request_error",
            message=f"Invalid argument: {str(e)}",
            code=None,
            param=None,
        )

    if isinstance(e, ValidationError):
        return DialException(
            status_code=422,
            type="invalid_request_error",
            message=e.message,
            code="invalid_argument",
            param=None,
        )

    if isinstance(e, DialException):
        return e

    return DialException(
        status_code=500,
        type="internal_server_error",
        message=str(e),
        code=None,
        param=None,
    )


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log.exception(e)
            raise to_dial_exception(e) from e

    return wrapper
