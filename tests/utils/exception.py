import contextlib
import re
from typing import List, overload

from openai import APIError
from pydantic import BaseModel


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str
    display_message: str | None = None
    status_code: int | None = None


@overload
async def expected_exception(
    exception: ExpectedException | List[ExpectedException],
): ...


@overload
async def expected_exception(
    cls: type[APIError],
    message: str,
    display_message: str | None = None,
    status_code: int | None = None,
): ...


@contextlib.asynccontextmanager
async def expected_exception(
    cls: type[APIError] | ExpectedException | List[ExpectedException],
    message: str | None = None,
    display_message: str | None = None,
    status_code: int | None = None,
):
    try:
        yield
    except Exception as e:
        if isinstance(cls, ExpectedException):
            exceptions = [cls]
        elif isinstance(cls, list):
            exceptions = cls
        else:
            assert message is not None
            exceptions = [
                ExpectedException(
                    type=cls,
                    message=message,
                    display_message=display_message,
                    status_code=status_code,
                )
            ]

        msgs = []
        for idx, exc in enumerate(exceptions, start=1):
            if (msg := _match_exception(e, exc)) is None:
                return
            msgs.append(f" [{idx}] {msg}")

        lines = "\n".join(msgs)
        assert (
            False
        ), f"The actual exception doesn't match actual exception:\n{lines}"
    else:
        assert False, f"The test didn't raise the expected exception {cls}"


def _match_exception(e: Exception, exc: ExpectedException) -> str | None:
    if not isinstance(e, exc.type):
        return f"Actual exception type ({type(e)}) doesn't match the expected one ({exc.type})"

    actual_status_code = getattr(e, "status_code", None)
    if actual_status_code != exc.status_code:
        return f"Actual status code ({actual_status_code}) doesn't match the expected one ({exc.status_code})"

    if not re.search(exc.message, str(e)):
        return f"The actual error message ({str(e)!r}) doesn't match the expected regexp ({exc.message!r})"

    actual_display_message = (e.body or {}).get("display_message")  # type: ignore
    if actual_display_message != exc.display_message:
        return f"Actual display message ({actual_display_message}) doesn't match the expected one ({exc.display_message})"

    return None
