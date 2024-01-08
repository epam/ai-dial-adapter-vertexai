from typing import Mapping

from aidial_sdk.chat_completion import Choice, Status
from aiohttp import hdrs

from aidial_adapter_vertexai.llm.exceptions import UserError


async def report_user_error(
    choice: Choice, headers: Mapping[str, str], error: UserError
) -> None:
    message = error.message
    is_chat_usage = headers.get(hdrs.AUTHORIZATION) is not None
    if is_chat_usage:
        stage = choice.create_stage("Error")
        stage.append_content(message)
        stage.close(Status.FAILED)
    else:
        raise Exception(message)
