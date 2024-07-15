from functools import wraps

from aidial_sdk import HTTPException as DialException
from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
)
from google.auth.exceptions import GoogleAuthError

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.utils.log_config import app_logger as log


def to_dial_exception(e: Exception) -> DialException:
    if isinstance(e, GoogleAuthError):
        return DialException(
            status_code=401,
            type="invalid_request_error",
            message=f"Invalid Authentication: {str(e)}",
            code="invalid_api_key",
        )

    if isinstance(e, PermissionDenied):
        return DialException(
            status_code=403,
            type="invalid_request_error",
            message=f"Permission denied: {str(e)}",
            code="permission_denied",
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
        )

    if isinstance(e, GoogleAPICallError):
        return DialException(
            status_code=e.code or 500,
            type="invalid_request_error",
            message=f"Invalid argument: {str(e)}",
        )

    if isinstance(e, ValidationError):
        return e.to_dial_exception()

    if isinstance(e, UserError):
        return e.to_dial_exception()

    if isinstance(e, DialException):
        return e

    return DialException(
        status_code=500,
        type="internal_server_error",
        message=str(e),
    )


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log.exception(
                f"caught exception: {type(e).__module__}.{type(e).__name__}"
            )
            raise to_dial_exception(e) from e

    return wrapper
