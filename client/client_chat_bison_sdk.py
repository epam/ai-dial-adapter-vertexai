import vertexai
from vertexai.preview.language_models import ChatModel, CodeChatModel

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from aidial_adapter_vertexai.utils.env import get_env
from client.conf import MAX_CHAT_TURNS, MAX_INPUT_CHARS
from client.utils.cli import select_enum, select_option
from client.utils.init import init
from client.utils.input import make_input
from client.utils.printing import print_ai


def main():
    location = get_env("DEFAULT_REGION")
    project = get_env("GCP_PROJECT_ID")

    model_id = select_enum("Select the model", ChatCompletionDeployment)
    streaming = select_option("Streaming?", [False, True])

    vertexai.init(project=project, location=location)

    match model_id:
        case ChatCompletionDeployment.CHAT_BISON_1:
            chat_model = ChatModel.from_pretrained(model_id)
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            chat_model = CodeChatModel.from_pretrained(model_id)

    parameters = {
        "temperature": 0.0,
        "max_output_tokens": 1024,
    }

    chat = chat_model.start_chat(**parameters)

    chat_input = make_input()

    turn = 0
    while turn < MAX_CHAT_TURNS:
        turn += 1
        content = chat_input()[:MAX_INPUT_CHARS]

        if streaming:
            responses = chat.send_message_streaming(content)
            for response in responses:
                print_ai(response.text, end="")
            print_ai("")
        else:
            response = chat.send_message(content)
            print_ai(response.text)


if __name__ == "__main__":
    init()
    main()
