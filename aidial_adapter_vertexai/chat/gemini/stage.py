from typing import List

from aidial_sdk.chat_completion import Stage
from google.genai.types import Language as GenAILanguage

from aidial_adapter_vertexai.chat.consumer import Consumer


class LazyStage:
    consumer: Consumer
    name: str
    stage: Stage | None = None

    def __init__(self, consumer: Consumer, name: str):
        self.consumer = consumer
        self.name = name

    async def append_content(self, content: str):
        if not self.stage:
            self.stage = await self.consumer.create_stage(self.name)
            self.stage.open()
        self.stage.append_content(content)

    def opened(self) -> bool:
        return self.stage is not None

    def close(self):
        if not self.stage:
            return
        self.stage.close()
        self.stage = None


class CodeExecutionStage:
    stage: LazyStage
    prev_lang: GenAILanguage | None = None
    first_code: bool = True
    outputs: List[str] = []

    def __init__(self, consumer: Consumer):
        self.stage = LazyStage(consumer, "Code execution")

    async def append_code(
        self, code: str, cur_lang: GenAILanguage | None
    ) -> None:
        block = "\n```\n"

        def to_code_block(lang: GenAILanguage | None):
            lang_str = "py" if lang == GenAILanguage.PYTHON else ""
            return f"\n```{lang_str}\n"

        content = ""
        if self.first_code:
            content = f"{to_code_block(cur_lang)}{code}"
        elif cur_lang != self.prev_lang:
            # block at the beginning to close prev code
            content = f"{block}{to_code_block(cur_lang)}{code}"
        else:
            content = code

        await self.stage.append_content(content)
        self.prev_lang = cur_lang
        self.first_code = False

    async def append_code_output(self, code_output: str) -> None:
        self.outputs.append(code_output)

    async def close(self) -> None:
        if not self.stage.opened():
            return

        block = "\n```\n"
        if len(self.outputs) == 0:
            # close the code
            await self.stage.append_content(block)
        else:
            content = (
                # block at the beginning to close prev code
                f"{block}"
                f"Code output"
                f"{block}"
                f"{''.join(self.outputs)}"
                f"{block}"
            )
            await self.stage.append_content(content)
        self.stage.close()
