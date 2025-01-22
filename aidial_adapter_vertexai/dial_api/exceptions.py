from functools import wraps

import anthropic
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError, InvalidRequestError
from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
)
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError

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
            return InvalidRequestError(
                message=content_filter_msg,
                code="content_filter",
                param="prompt",
            )

        return InvalidRequestError(
            f"Invalid argument: {str(e)}",
        )

    if isinstance(e, (GoogleAPICallError, APIError)):
        code = e.code or 500
        return DialException(
            status_code=code,
            type=(
                "invalid_request_error"
                if code < 500
                else "internal_server_error"
            ),
            message=str(e),
        )

    if isinstance(e, anthropic.APIStatusError):
        code = e.status_code
        try:
            response = e.response.json()["error"]
            return DialException(
                status_code=code,
                type=response["type"],
                message=response["message"],
            )
        except Exception:
            return DialException(
                status_code=code,
                type=(
                    "invalid_request_error"
                    if code < 500
                    else "internal_server_error"
                ),
                message=e.message,
            )

    if isinstance(e, ValidationError):
        return e.to_dial_exception()

    if isinstance(e, UserError):
        return e.to_dial_exception()

    if isinstance(e, DialException):
        return e

    return InternalServerError(str(e))


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            dial_exception = to_dial_exception(e)
            log.exception(
                f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
                f"The exception converted to the dial exception: {dial_exception!r}."
            )
            raise dial_exception from e

    return wrapper
