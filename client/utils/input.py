from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from client.utils.files import get_project_root


def make_input(max_input_chars: int = 1024):
    session = None

    def input(prompt_text="> ", style=Style.from_dict({"": "#ff0000"})) -> str:
        nonlocal session
        if session is None:
            session = PromptSession(
                history=FileHistory(str(get_project_root() / ".history"))
            )

        response = session.prompt(prompt_text, style=style, in_thread=True)
        return response[:max_input_chars]

    return input
