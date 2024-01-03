import asyncio
import logging.config
from typing import Tuple, assert_never

from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.universal_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.env import get_env
from aidial_adapter_vertexai.utils.log_config import LogConfig
from aidial_adapter_vertexai.utils.timer import Timer
from client.chat import AdapterChat, Chat, create_sdk_chat
from client.conf import MAX_CHAT_TURNS, MAX_INPUT_CHARS
from client.config import ClientMode, Config
from client.utils.input import make_input
from client.utils.printing import print_ai, print_info

logging.config.dictConfig(LogConfig().dict())


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

    turn = 0
    while turn < MAX_CHAT_TURNS:
        turn += 1

        content = input()[:MAX_INPUT_CHARS]

        if content in [":q", ":quit"]:
            break
        elif content in [":r", ":restart"]:
            chat, model_parameters = await init_chat(Config.get_interactive())
            continue
        elif content == "":
            continue

        usage = TokenUsage()

        timer = Timer()

        async for chunk in chat.send_message(content, model_parameters, usage):
            print_ai(chunk, end="")

        print_ai("")

        print_info(f"Timing: {timer}")
        print_info(f"Usage: {usage}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_info("Shutting down...")
