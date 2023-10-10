import asyncio
from typing import List

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_vertexai.llm.vertex_ai_adapter import (
    get_chat_completion_model,
)
from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.universal_api.request import ModelParameters
from aidial_adapter_vertexai.utils.env import get_env
from client.conf import MAX_CHAT_TURNS, MAX_INPUT_CHARS
from client.utils.cli import select_enum, select_option
from client.utils.init import init
from client.utils.input import make_input
from client.utils.printing import print_ai, print_info


async def main():
    location = get_env("DEFAULT_REGION")
    project = get_env("GCP_PROJECT_ID")

    model_id = select_enum("Select the model", ChatCompletionDeployment)
    streaming = select_option("Streaming?", [False, True])

    model = await get_chat_completion_model(
        location=location,
        project_id=project,
        deployment=model_id,
        model_params=ModelParameters(),
    )

    history: List[Message] = []

    input = make_input()

    turn = 0
    while turn < MAX_CHAT_TURNS:
        turn += 1

        content = input()[:MAX_INPUT_CHARS]
        history.append(Message(role=Role.USER, content=content))

        response, usage = await model.chat(streaming, history)

        print_info(usage.json(indent=2))

        print_ai(response.strip())
        history.append(Message(role=Role.ASSISTANT, content=response))


if __name__ == "__main__":
    init()
    asyncio.run(main())
