from typing import Mapping

from aidial_sdk.chat_completion import Choice, Status
from aiohttp import hdrs

from aidial_adapter_vertexai.llm.exceptions import UserError


def report_user_error(
    choice: Choice, headers: Mapping[str, str], error: UserError
) -> None:
    is_chat_usage = headers.get(hdrs.AUTHORIZATION) is not None
    if is_chat_usage:
        add_error_stage(choice, error.to_message_for_chat_user())
    else:
        raise Exception(error.message)


def add_error_stage(choice: Choice, message: str) -> None:
    stage = choice.create_stage("Error")
    stage.open()
    stage.append_content(message)
    stage.close(Status.FAILED)
