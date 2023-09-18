from typing import Callable, List

from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def sanitize_test_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())


async def assert_dialog(
    model: BaseChatModel,
    messages: List[BaseMessage],
    output_predicate: Callable[[str], bool],
    streaming: bool,
):
    llm_result = await model.agenerate([messages])

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage in [None, {}]) == streaming

    actual_output = llm_result.generations[0][-1].text

    assert output_predicate(
        actual_output
    ), f"Failed output test, actual output: {actual_output}"
