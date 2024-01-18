import re
from typing import Callable, List, Optional

from langchain.callbacks.base import Callbacks
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from tests.conftest import DEFAULT_API_VERSION
from tests.utils.callback import CallbackWithNewLines


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def sanitize_test_name(name: str) -> str:
    name2 = "".join(c if c.isalnum() else "_" for c in name.lower())
    return re.sub("_+", "_", name2)


def for_all(
    predicate: Callable[[str], bool], n: int = 1
) -> Callable[[List[str]], bool]:
    def f(candidates: List[str]) -> bool:
        assert (
            len(candidates) == n
        ), f"Expected {n} candidates, got {len(candidates)}"
        return all(predicate(output) for output in candidates)

    return f


async def assert_dialog(
    model: BaseChatModel,
    messages: List[BaseMessage],
    output_predicate: Callable[[List[str]], bool],
    streaming: bool,
    stop: Optional[List[str]] = None,
    n: Optional[int] = None,
):
    llm_result = await model.agenerate([messages], stop=stop, n=n)

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage in [None, {}]) == streaming

    print(llm_result)

    actual_output = [candidate.text for candidate in llm_result.generations[-1]]

    assert output_predicate(
        actual_output
    ), f"Failed output test, actual output: {actual_output}"


def create_chat_model(
    base_url: str,
    model_id: str,
    streaming: bool,
    max_tokens: Optional[int],
) -> BaseChatModel:
    callbacks: Callbacks = [CallbackWithNewLines()]
    return AzureChatOpenAI(
        deployment_name=model_id,
        callbacks=callbacks,
        openai_api_base=base_url,
        openai_api_version=DEFAULT_API_VERSION,
        openai_api_key="dummy_key",
        verbose=True,
        streaming=streaming,
        temperature=0,
        request_timeout=10,
        max_retries=0,
        max_tokens=max_tokens,
        client=None,
    )
