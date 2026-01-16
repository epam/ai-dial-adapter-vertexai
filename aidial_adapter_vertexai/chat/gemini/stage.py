from typing import List

from aidial_sdk.chat_completion import Stage
from google.genai.types import Language as GenAILanguage

from aidial_adapter_vertexai.chat.consumer import Consumer


class LazyStage:
    _consumer: Consumer
    _name: str
    _stage: Stage | None = None

    def __init__(self, consumer: Consumer, name: str):
        self._consumer = consumer
        self._name = name

    async def append_content(self, content: str):
        if not self._stage:
            self._stage = await self._consumer.create_stage(self._name)
            self._stage.open()
        self._stage.append_content(content)

    def __bool__(self):
        return self._stage is not None

    def close(self):
        if self._stage:
            self._stage.close()
            self._stage = None


class CodeExecutionStage:
    _stage: LazyStage
    _outputs: List[str]
    _prev_lang: GenAILanguage | None = None
    _first_code: bool = True

    def __init__(self, consumer: Consumer, name: str):
        self._stage = LazyStage(consumer, name)
        self._outputs = []

    async def append_code(self, code: str, lang: GenAILanguage | None) -> None:
        block = "```\n"
        lang_str = "py" if lang == GenAILanguage.PYTHON else ""
        lang_block = f"```{lang_str}\n"
        content = ""

        if self._first_code:
            content = f"{lang_block}{code}"
        elif lang != self._prev_lang:
            # block at the beginning to close prev code
            content = f"{block}{lang_block}{code}"
        else:
            content = code

        await self._stage.append_content(content)
        self._prev_lang = lang
        self._first_code = False

    async def append_code_output(self, code_output: str) -> None:
        self._outputs.append(code_output)

    async def close(self) -> None:
        if not self._stage:
            return

        ticks = "```"
        # close the code
        content = f"\n{ticks}"
        if len(self._outputs):
            content += (
                "\nCode output"
                f"{ticks}\n"
                f"{''.join(self._outputs)}"
                f"{ticks}"
            )
        await self._stage.append_content(content)
        self._stage.close()
