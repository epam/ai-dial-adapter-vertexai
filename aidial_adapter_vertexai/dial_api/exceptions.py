from functools import wraps

import anthropic
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError

from aidial_adapter_vertexai.chat.errors import UserError, ValidationError
from aidial_adapter_vertexai.utils.log_config import app_logger as log


def _get_exception_type(code: int) -> str:
    return "invalid_request_error" if code < 500 else "internal_server_error"


_CONTENT_FILTER_MSG = "The response is blocked, as it may violate our policies."


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

    if isinstance(e, (GoogleAPICallError, APIError)):
        status_code = e.code or 500
        message = e.message or str(e)

        code = None
        if _CONTENT_FILTER_MSG in message:
            message = _CONTENT_FILTER_MSG
            code = "content_filter"

        return DialException(
            status_code=status_code,
            type=_get_exception_type(status_code),
            code=code,
            message=message,
        )

    if isinstance(e, anthropic.APIStatusError):
        try:
            message = e.body["error"]["message"]  # type: ignore
        except Exception:
            message = e.message

        code = e.status_code
        # Strangely, Anthropic returns 200 status code with the Overloaded exception.
        if code == 200 and message == "Overloaded":
            code = 503

        return DialException(
            status_code=code,
            type=_get_exception_type(code),
            message=message,
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
