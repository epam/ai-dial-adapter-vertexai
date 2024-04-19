import asyncio
from pathlib import Path
from typing import List, Tuple, assert_never

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.chat.gemini.inputs import MessageWithResources
from aidial_adapter_vertexai.dial_api.request import ModelParameters
from aidial_adapter_vertexai.dial_api.token_usage import TokenUsage
from aidial_adapter_vertexai.utils.env import get_env
from aidial_adapter_vertexai.utils.log_config import configure_loggers
from aidial_adapter_vertexai.utils.resource import Resource
from aidial_adapter_vertexai.utils.timer import Timer
from client.chat.adapter import AdapterChat
from client.chat.base import Chat
from client.chat.sdk import create_sdk_chat
from client.conf import MAX_CHAT_TURNS, MAX_INPUT_CHARS
from client.config import ClientMode, Config
from client.utils.input import make_input
from client.utils.printing import print_ai, print_error, print_info

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

    turn = 0
    while turn < MAX_CHAT_TURNS:
        turn += 1

        query = input()[:MAX_INPUT_CHARS]

        if query in [":q", ":quit"]:
            break
        elif query in [":r", ":restart"]:
            chat, model_parameters = await init_chat(Config.get_interactive())
            continue
        elif any(query.startswith(cmd) for cmd in [":a ", ":attach "]):
            path = Path(query.split(" ", 1)[1])
            resources.append(Resource.from_path(path))
            continue
        elif query == "":
            continue

        usage = TokenUsage()
        timer = Timer()

        message = Message(role=Role.USER, content=query)

        try:
            async for chunk in chat.send_message(
                MessageWithResources(message=message, resources=resources),
                model_parameters,
                usage,
            ):
                print_ai(chunk, end="")

            print_ai("")
        except Exception as e:
            print_error(f"Error: {str(e)}")

        resources = []

        print_info(f"Timing: {timer}")
        print_info(f"Usage: {usage}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_info("Shutting down...")
