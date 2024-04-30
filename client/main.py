import asyncio
from pathlib import Path
from typing import List, Tuple, assert_never

from aidial_sdk.chat_completion import CustomContent, Function, Message, Role

from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithResources
from aidial_adapter_vertexai.chat.tools import ToolsConfig
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.env import get_env
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.log_config import configure_loggers
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.timer import Timer
from client.chat.adapter import AdapterChat
from client.chat.base import Chat
from client.chat.sdk import create_sdk_chat
from client.conf import MAX_CHAT_TURNS, MAX_INPUT_CHARS
from client.config import ClientMode, Config
from client.utils.input import make_input
from client.utils.printing import print_ai

configure_loggers()


async def init_chat(params: Config) -> Tuple[Chat, ModelParameters]:
    location = get_env("DEFAULT_REGION")
    project = get_env("GCP_PROJECT_ID")

    chat: Chat
    match params.mode:
        case ClientMode.ADAPTER:
            chat = await AdapterChat.create(location, project, params.model_id)
        case ClientMode.SDK:
            chat = await create_sdk_chat(location, project, params.model_id)
        case _:
            assert_never(params.mode)

    return chat, params.to_model_parameters()


async def main():
    chat, model_parameters = await init_chat(Config.get_interactive())

    input = make_input()

    resources: List[Resource] = []
    functions: List[Function] = []

    turn = 0
    while turn < MAX_CHAT_TURNS:
        turn += 1

        query = input()[:MAX_INPUT_CHARS]

        if query == "":
            continue

        if query in [":q", ":quit"]:
            break

        if query in [":r", ":restart"]:
            chat, model_parameters = await init_chat(Config.get_interactive())
            continue

        if any(query.startswith(cmd) for cmd in [":a ", ":attach "]):
            path = Path(query.split(" ", 1)[1])
            try:
                resources.append(Resource.from_path(path))
            except Exception as e:
                log.error(f"Can't load Resource: {str(e)}")
            continue

        if any(query.startswith(cmd) for cmd in [":f ", ":func "]):
            decl = query.split(" ", 1)[1]
            try:
                functions.append(Function.parse_raw(decl))
            except Exception as e:
                log.error(f"Can't parse Function: {str(e)}")
            continue

        if any(query.startswith(cmd) for cmd in [":message "]):
            resp = query.split(" ", 1)[1]
            try:
                message = Message.parse_raw(resp)
            except Exception as e:
                log.error(f"Can't parse Message: {str(e)}")
                continue
        else:
            message = Message(role=Role.USER, content=query)

        attachments = [res.to_attachment() for res in resources]
        message.custom_content = CustomContent(attachments=attachments)

        usage = TokenUsage()
        timer = Timer()

        try:
            async for chunk in chat.send_message(
                ToolsConfig(functions=functions, is_tool=True),
                MessageWithResources(message=message, resources=resources),
                model_parameters,
                usage,
            ):
                print_ai(chunk, end="")

            print_ai("")
        except Exception as e:
            log.exception(e)

        resources = []

        log.info(f"Timing: {timer}")
        log.info(f"Usage: {usage}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
